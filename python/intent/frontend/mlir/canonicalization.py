from __future__ import annotations

from dataclasses import replace

from mlir import ir

from ..semantics.operations import ShapeRelation
from ..semantics.types import RecordType, TensorType, TupleType
from .attributes import emit_dictionary
from .types import emit_type


_PROVENANCE = {
    "intent.node", "intent.result_nodes", "intent.result_names",
    "intent.region_argument_nodes", "intent.region_argument_names",
}
_ELEMENTWISE = {
    "intent.constant", "intent.dim", "intent.broadcast", "intent.full",
    "intent.cast", "intent.bitcast", "intent.unary", "intent.binary",
    "intent.compare", "intent.select", "intent.mask", "intent.make_tuple",
    "intent.make_record", "intent.extract", "intent.yield",
}


def _walk(operation):
    for region in operation.regions:
        for block in region.blocks:
            for nested in list(block.operations):
                yield from _walk(nested.operation)
    yield operation


def _attributes(operation):
    return {item.name: item.attr for item in operation.attributes
            if item.name not in _PROVENANCE}


def _element_type(value_type):
    return (ir.RankedTensorType(value_type).element_type
            if ir.RankedTensorType.isinstance(value_type) else value_type)


def _strip_scalar_broadcast(value):
    while isinstance(value.owner, ir.Operation):
        operation = value.owner
        if operation.name not in ("intent.broadcast", "intent.full"):
            break
        source = operation.operands[0]
        if (ir.RankedTensorType.isinstance(source.type)
                and ir.RankedTensorType(source.type).rank != 0):
            break
        value = source
    return value


def _constant_attribute(value):
    value = _strip_scalar_broadcast(value)
    if not isinstance(value.owner, ir.Operation):
        return None
    operation = value.owner
    if operation.name == "intent.constant":
        source = operation.attributes["value"]
    elif operation.name == "intent.cast":
        source = _constant_attribute(operation.operands[0])
    else:
        return None
    if source is None:
        return None
    destination = _element_type(value.type)
    if ir.FloatAttr.isinstance(source):
        number = ir.FloatAttr(source).value
    elif ir.IntegerAttr.isinstance(source):
        number = ir.IntegerAttr(source).value
    else:
        return None
    if (ir.F16Type.isinstance(destination) or ir.BF16Type.isinstance(destination)
            or ir.F32Type.isinstance(destination) or ir.F64Type.isinstance(destination)):
        with operation.location:
            return ir.FloatAttr.get(destination, float(number))
    if ir.IntegerType.isinstance(destination) and isinstance(number, int):
        return ir.IntegerAttr.get(destination, number)
    return None


def _resolve(reference):
    value, path = reference
    while isinstance(value.owner, ir.Operation):
        operation = value.owner
        if operation.name == "intent.extract":
            path = (ir.IntegerAttr(operation.attributes["field"]).value,) + path
            value = operation.operands[0]
        elif path and operation.name in ("intent.make_record", "intent.make_tuple"):
            value, path = operation.operands[path[0]], path[1:]
        else:
            break
    return value, path


def _leaves(value, types):
    def flatten(value_type, path):
        if isinstance(value_type, RecordType):
            fields = tuple(field for _, field in value_type.fields)
        elif isinstance(value_type, TupleType):
            fields = value_type.components
        else:
            return [(_resolve((value, path)), value_type)]
        return [leaf for index, field in enumerate(fields)
                for leaf in flatten(field, path + (index,))]
    return flatten(types[value.type], ())


def _flatten(values, types):
    return [reference for value in values for reference, _ in _leaves(value, types)]


def _same_reference(lhs, rhs, mapping):
    lhs, rhs = _resolve(lhs), _resolve(rhs)
    value, path = lhs
    for count in range(len(path), -1, -1):
        key = value, path[:count]
        if key in mapping:
            target, prefix = mapping[key]
            return _resolve((target, prefix + path[count:])) == rhs
    if lhs == rhs:
        return True
    if lhs[1] or rhs[1]:
        return False
    lhs, rhs = lhs[0], rhs[0]
    left_constant, right_constant = _constant_attribute(lhs), _constant_attribute(rhs)
    if left_constant is not None and right_constant is not None:
        return left_constant == right_constant
    lhs, rhs = _strip_scalar_broadcast(lhs), _strip_scalar_broadcast(rhs)
    if (lhs, ()) in mapping:
        return mapping[(lhs, ())] == (rhs, ())
    if lhs == rhs:
        return True
    if (_element_type(lhs.type) != _element_type(rhs.type)
            or not isinstance(lhs.owner, ir.Operation)
            or not isinstance(rhs.owner, ir.Operation)):
        return False
    left, right = lhs.owner, rhs.owner
    if (left.name not in _ELEMENTWISE or left.name != right.name
            or _attributes(left) != _attributes(right)
            or len(left.operands) != len(right.operands)):
        return False
    return all(_same_reference((a, ()), (b, ()), mapping)
               for a, b in zip(left.operands, right.operands))


def _same_combine(lhs, rhs, types):
    left, right = lhs.blocks[0], rhs.blocks[0]
    left_args = [leaf for value in left.arguments for leaf in _leaves(value, types)]
    right_args = [leaf for value in right.arguments for leaf in _leaves(value, types)]
    if len(left_args) != len(right_args):
        return False
    if any(a.dtype != b.dtype for (_, a), (_, b) in zip(left_args, right_args)):
        return False
    mapping = {a: b for (a, _), (b, _) in zip(left_args, right_args)}
    left_yield = _flatten(list(left.operations)[-1].operands, types)
    right_yield = _flatten(list(right.operations)[-1].operands, types)
    return (len(left_yield) == len(right_yield)
            and all(_same_reference(a, b, mapping)
                    for a, b in zip(left_yield, right_yield)))


def _elementwise_body(body, collective):
    for operation in body.operations:
        if operation.name not in _ELEMENTWISE | {collective}:
            return False
        if operation.name == "intent.dim":
            for use in operation.results[0].uses:
                if (use.owner.name not in ("intent.broadcast", "intent.full")
                        or use.operand_number == 0):
                    return False
    return True


def _summary_reduce(operation, types):
    body = operation.regions[0].blocks[0]
    reductions = [nested.operation for nested in body.operations
                  if nested.name == "intent.reduce"]
    if len(reductions) != 1:
        return None
    reduction = reductions[0]
    if not _elementwise_body(body, "intent.reduce"):
        return None
    if _flatten(list(body.operations)[-1].operands, types) != _flatten(reduction.results, types):
        return None
    axes = tuple(ir.IntegerAttr(axis).value
                 for axis in ir.ArrayAttr(reduction.attributes["axes"]))
    if axes != (ir.IntegerAttr(operation.attributes["axis"]).value,):
        return None
    if not _same_combine(operation.regions[1], reduction.regions[0], types):
        return None
    count = ir.IntegerAttr(operation.attributes["source_count"]).value
    identities = ir.IntegerAttr(operation.attributes["identity_count"]).value
    reduce_sources = ir.IntegerAttr(reduction.attributes["source_count"]).value
    reduce_identities = ir.IntegerAttr(reduction.attributes["identity_count"]).value
    captures_start = count + identities
    if operation.name == "intent.region_scan":
        captures_start += ir.IntegerAttr(operation.attributes["state_count"]).value
    mapping = {(a, ()): (b, ()) for a, b in zip(
        body.arguments, list(operation.operands[:count])
        + list(operation.operands[captures_start:]))}
    left_identity = _flatten(reduction.operands[
        reduce_sources:reduce_sources + reduce_identities], types)
    right_identity = _flatten(operation.operands[count:count + identities], types)
    if (len(left_identity) != len(right_identity)
            or not all(_same_reference(a, b, mapping)
                       for a, b in zip(left_identity, right_identity))):
        return None
    return reduction


def _element_scan(operation, reduction, types):
    body = operation.regions[3].blocks[0]
    scans = [nested.operation for nested in body.operations
             if nested.name == "intent.scan"]
    if len(scans) != 1 or not _elementwise_body(body, "intent.scan"):
        return None
    scan = scans[0]
    if (scan.attributes["axis"] != operation.attributes["axis"]
            or not ir.BoolAttr(scan.attributes["inclusive"]).value
            or ir.BoolAttr(scan.attributes["reverse"]).value):
        return None
    if not _same_combine(operation.regions[1], scan.regions[0], types):
        return None
    count = ir.IntegerAttr(operation.attributes["source_count"]).value
    states = ir.IntegerAttr(operation.attributes["state_count"]).value
    summary_args = operation.regions[0].blocks[0].arguments
    mapping = {(a, ()): (b, ()) for a, b in zip(
        summary_args, list(body.arguments[:count]) + list(body.arguments[count + states:]))}
    if len(reduction.operands) != len(scan.operands):
        return None
    if not all(_same_reference((a, ()), (b, ()), mapping)
               for a, b in zip(reduction.operands, scan.operands)):
        return None
    if any(nested.name not in _ELEMENTWISE
           for nested in operation.regions[2].blocks[0].operations):
        return None
    return scan


def _replace_dimensions(value_type, substitutions):
    if isinstance(value_type, TensorType):
        return TensorType(value_type.dtype, tuple(substitutions.get(dim, dim)
                                                  for dim in value_type.shape))
    if isinstance(value_type, RecordType):
        return RecordType(tuple((name, _replace_dimensions(field, substitutions))
                                for name, field in value_type.fields))
    if isinstance(value_type, TupleType):
        return TupleType(tuple(_replace_dimensions(field, substitutions)
                               for field in value_type.components))
    return value_type


def canonicalize_regions(module, builder):
    regions = [operation for operation in _walk(module.operation)
               if operation.name in ("intent.region_fold", "intent.region_scan")]
    if not regions:
        return
    types = {ir.Type.parse(emit_type(value.type, builder.dimension_id)): value.type
             for value in builder._values if value.view_access is None}

    def clone_body(region, arguments, owner, substitutions):
        body = region.blocks[0]
        mapping = dict(zip(body.arguments, arguments))
        dimension_ids = {builder.dimension_id(old): builder.dimension_id(new)
                         for old, new in substitutions.items()}

        def remap_type(value_type):
            descriptor = types[value_type]
            rewritten = _replace_dimensions(descriptor, substitutions)
            result = ir.Type.parse(emit_type(rewritten, builder.dimension_id))
            types[result] = rewritten
            return result

        def remap(original, cloned):
            for index, operand in enumerate(original.operands):
                cloned.operands[index] = mapping.get(operand, operand)
            for old, new in zip(original.results, cloned.results):
                new.set_type(remap_type(old.type))
                mapping[old] = new
            node = ir.IntegerAttr(original.attributes["intent.node"]).value
            if original.name == "intent.dim":
                dimension = ir.IntegerAttr(original.attributes["dimension"])
                cloned.attributes["dimension"] = ir.IntegerAttr.get(
                    dimension.type, dimension_ids.get(dimension.value, dimension.value))
            shape = builder._shape_relations.get(node)
            if shape is not None:
                rewritten = ShapeRelation(tuple(replace(axis, dimension=dimension_ids.get(
                    axis.dimension, axis.dimension)) for axis in shape.axes))
                cloned.attributes["shape"] = ir.DictAttr(ir.Attribute.parse(
                    emit_dictionary({"shape": rewritten})))["shape"]
            for old_region, new_region in zip(original.regions, cloned.regions):
                for old_block, new_block in zip(old_region.blocks, new_region.blocks):
                    for old, new in zip(old_block.arguments, new_block.arguments):
                        new.set_type(remap_type(old.type))
                        mapping[old] = new
                    for old, new in zip(old_block.operations, new_block.operations):
                        remap(old.operation, new.operation)

        # Each body is moved once; erasing its outer region retires the old IDs.
        operations = list(body.operations)
        for operation in operations[:-1]:
            cloned = operation.clone(ip=ir.InsertionPoint(owner)).operation
            remap(operation.operation, cloned)
            if cloned.name == "intent.extract":
                value, path = _resolve((cloned.results[0], ()))
                if not path and value != cloned.results[0]:
                    cloned.results[0].replace_all_uses_with(value)
                    mapping[operation.results[0]] = value
                    cloned.erase()
        return [mapping.get(value, value) for value in operations[-1].operands]

    for operation in regions:
        reduction = _summary_reduce(operation, types)
        if reduction is None:
            continue
        scan = operation.name == "intent.region_scan"
        if scan and _element_scan(operation, reduction, types) is None:
            continue
        count = ir.IntegerAttr(operation.attributes["source_count"]).value
        identities = ir.IntegerAttr(operation.attributes["identity_count"]).value
        states = ir.IntegerAttr(operation.attributes["state_count"]).value if scan else 0
        sources = list(operation.operands[:count])
        initial = list(operation.operands[count + identities:count + identities + states])
        captures = list(operation.operands[count + identities + states:])
        axis = ir.IntegerAttr(operation.attributes["axis"]).value
        source_type = types[sources[0].type]
        slice_type = types[operation.regions[0].blocks[0].arguments[0].type]
        substitutions = {slice_type.shape[axis]: source_type.shape[axis]}
        summary = clone_body(operation.regions[0], sources + captures,
                             operation, substitutions)
        if scan:
            output = clone_body(operation.regions[3], sources + initial + captures,
                                operation, substitutions)
            final = clone_body(operation.regions[2], summary + initial,
                               operation, substitutions)
            replacement = output + final
        else:
            replacement = summary
        for old, new in zip(operation.results, replacement):
            old.replace_all_uses_with(new)
        operation.erase()
