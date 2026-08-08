#ifndef INTENT_DIALECT_INTENT_IR_INTENTTYPES_TD
#define INTENT_DIALECT_INTENT_IR_INTENTTYPES_TD

include "Intent/Dialect/Intent/IR/IntentDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Intent_LogicalType<string cppName, string typeMnemonic, string typeSummary>
    : TypeDef<Intent_Dialect, cppName> {
  let mnemonic = typeMnemonic;
  let summary = typeSummary;
  let parameters = (ins StringRefParameter<"canonical Kernel IR type spelling">:$spec);
  let assemblyFormat = "`<` $spec `>`";
}

def Intent_LogicalIndexType : Intent_LogicalType<
    "LogicalIndex", "logical_index", "An index in a named logical relation">;
def Intent_DomainType : Intent_LogicalType<
    "Domain", "domain", "A logical iteration domain">;
def Intent_RegionType : Intent_LogicalType<
    "Region", "region", "A logical subregion of a domain">;
def Intent_PartitionType : Intent_LogicalType<
    "Partition", "partition", "A partitioned logical region">;
def Intent_RaggedType : Intent_LogicalType<
    "Ragged", "ragged", "A ragged outer/member relation">;
def Intent_BufferType : Intent_LogicalType<
    "Buffer", "buffer", "A logical kernel-local buffer">;
def Intent_RecordType : Intent_LogicalType<
    "Record", "record", "A named-field logical record">;
def Intent_ConstexprType : Intent_LogicalType<
    "Constexpr", "constexpr", "A compile-time kernel parameter">;
def Intent_EnumType : Intent_LogicalType<
    "Enum", "enum", "A closed compile-time enumeration">;
def Intent_ViewType : TypeDef<Intent_Dialect, "View"> {
  let mnemonic = "view";
  let summary = "An external strided kernel view with an explicit access mode";
  let parameters = (ins
    TypeParameter<"::mlir::Type", "ranked tensor element and shape schema">:$tensor,
    StringRefParameter<"Intent view access mode">:$access
  );
  let assemblyFormat = "`<` $tensor `,` $access `>`";
}

#endif
