#include "Intent/Dialect/GPU/Transforms/TuningProfiles.h"
#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace mlir;

namespace intent::gpu {
namespace {

FailureOr<llvm::json::Value> readJSON(Location location, StringRef filename) {
  auto buffer = llvm::MemoryBuffer::getFile(filename);
  if (!buffer) {
    emitError(location) << "cannot read tuning profiles '" << filename << "': "
                        << buffer.getError().message();
    return failure();
  }
  auto value = llvm::json::parse((*buffer)->getBuffer());
  if (!value) {
    emitError(location) << "invalid tuning JSON '" << filename << "': "
                        << llvm::toString(value.takeError());
    return failure();
  }
  return std::move(*value);
}

FailureOr<ArrayAttr> readTable(Location location,
                              const llvm::json::Value &value, StringRef family) {
  auto array = value.getAsArray();
  if (!array) {
    emitError(location) << "tuning profile '" << family << "' needs a row array";
    return failure();
  }
  Builder builder(location.getContext());
  SmallVector<Attribute> rows;
  for (const llvm::json::Value &value : *array) {
    auto columns = value.getAsArray();
    if (!columns) {
      emitError(location) << "tuning profile '" << family << "' requires integer row arrays";
      return failure();
    }
    SmallVector<int64_t> row;
    for (const llvm::json::Value &column : *columns) {
      auto number = column.getAsInteger();
      if (!number) {
        emitError(location) << "tuning profile '" << family << "' values must be integers";
        return failure();
      }
      row.push_back(*number);
    }
    rows.push_back(builder.getDenseI64ArrayAttr(row));
  }
  return builder.getArrayAttr(rows);
}

LogicalResult checkColumns(const TuningProfileSchema &schema, ArrayAttr columns,
                           Location location) {
  if (columns.size() != schema.columns.size() ||
      !llvm::equal(columns, schema.columns, [](Attribute column, StringRef expected) {
        auto name = dyn_cast<StringAttr>(column);
        return name && name.getValue() == expected;
      }))
    return emitError(location) << "tuning column name/order disagrees for " << schema.space;
  return success();
}

} // namespace

FailureOr<TuningProfiles> TuningProfiles::read(
    Location location, ArrayRef<TuningProfileSource> defaults,
    StringRef overrideFilename) {
  Builder builder(location.getContext());
  NamedAttrList spaces;
  for (const TuningProfileSource &source : defaults) {
    auto parsed = readJSON(location, source.filename);
    if (failed(parsed))
      return failure();
    auto root = parsed->getAsObject();
    if (!root || root->size() != 2 || !root->getArray("columns") ||
        !root->getObject("families")) {
      emitError(location) << "default tuning profiles require only columns and families: "
                          << source.filename;
      return failure();
    }
    SmallVector<Attribute> columns;
    for (const llvm::json::Value &column : *root->getArray("columns")) {
      auto name = column.getAsString();
      if (!name) {
        emitError(location) << "tuning column names must be strings: " << source.filename;
        return failure();
      }
      columns.push_back(builder.getStringAttr(*name));
    }
    auto encodedColumns = builder.getArrayAttr(columns);
    if (failed(checkColumns(source.schema, encodedColumns, location)))
      return failure();
    NamedAttrList encodedFamilies;
    auto families = root->getObject("families");
    for (const auto &entry : *families) {
      auto table = readTable(location, entry.second, entry.first);
      if (failed(table))
        return failure();
      encodedFamilies.append(StringRef(entry.first), *table);
    }
    auto table = TuningProfileTableAttr::getChecked(location, builder.getContext(),
        encodedColumns, encodedFamilies.getDictionary(builder.getContext()));
    if (!table) return failure();
    if (spaces.get(source.schema.space)) {
      emitError(location) << "duplicate tuning namespace source '" << source.schema.space << "'";
      return failure();
    }
    spaces.append(source.schema.space, table);
  }
  if (!overrideFilename.empty()) {
    auto parsed = readJSON(location, overrideFilename);
    if (failed(parsed))
      return failure();
    auto root = parsed->getAsObject();
    if (!root) {
      emitError(location) << "tuning override requires a namespace object";
      return failure();
    }
    for (const auto &entry : *root) {
      StringRef name = entry.first;
      auto current = dyn_cast_or_null<TuningProfileTableAttr>(spaces.get(name));
      auto families = entry.second.getAsObject();
      if (!current || !families) {
        emitError(location) << "unknown or malformed tuning namespace '" << name << "'";
        return failure();
      }
      NamedAttrList updated(current.getFamilies());
      for (const auto &family : *families) {
        StringRef familyName = family.first;
        auto existing = current.getFamilies().get(familyName);
        if (!existing) {
          emitError(location) << "unknown tuning family '" << name << "." << familyName << "'";
          return failure();
        }
        auto table = readTable(location, family.second, familyName);
        if (failed(table))
          return failure();
        updated.set(familyName, *table);
      }
      auto table = TuningProfileTableAttr::getChecked(location, builder.getContext(),
          current.getColumns(), updated.getDictionary(builder.getContext()));
      if (!table) return failure();
      spaces.set(name, table);
    }
  }
  auto resolved = TuningProfilesAttr::getChecked(location,
      builder.getContext(), spaces.getDictionary(builder.getContext()));
  if (!resolved) return failure();
  return TuningProfiles(resolved);
}

FailureOr<TuningProfiles> TuningProfiles::from(ModuleOp module) {
  auto resolved = module->getAttrOfType<TuningProfilesAttr>(tuningProfilesAttr);
  if (!resolved)
    return module.emitError("GPU compilation requires resolved tuning profiles on the current module");
  return TuningProfiles(resolved);
}

void TuningProfiles::attach(ModuleOp module) const {
  module->setAttr(tuningProfilesAttr, profiles);
}

FailureOr<TuningProfiles::Table>
TuningProfiles::get(const TuningProfileSchema &schema, StringRef family,
                    Location location) const {
  auto space = profiles.getSpaces().getAs<TuningProfileTableAttr>(schema.space);
  if (space && failed(checkColumns(schema, space.getColumns(), location)))
    return failure();
  auto rows = space ? space.getFamilies().getAs<ArrayAttr>(family) : ArrayAttr();
  if (rows) {
    Table table;
    for (Attribute row : rows) {
      auto values = cast<DenseI64ArrayAttr>(row).asArrayRef();
      table.emplace_back(values.begin(), values.end());
    }
    return table;
  }
  emitError(location) << "missing declared tuning family '" << schema.space << "." << family << "'";
  return failure();
}

} // namespace intent::gpu
