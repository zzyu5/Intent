#include "Intent/Dialect/GPU/Transforms/TuningProfiles.h"

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

FailureOr<TuningProfiles::Table> readTable(Location location,
                                          const llvm::json::Value &value,
                                          size_t width, StringRef family) {
  auto array = value.getAsArray();
  if (!array || array->empty()) {
    emitError(location) << "tuning profile '" << family << "' needs a non-empty row array";
    return failure();
  }
  TuningProfiles::Table table;
  for (const llvm::json::Value &value : *array) {
    auto columns = value.getAsArray();
    if (!columns || columns->size() != width) {
      emitError(location) << "tuning profile '" << family << "' requires " << width
                          << " integer columns per row";
      return failure();
    }
    TuningProfiles::Row row;
    for (const llvm::json::Value &column : *columns) {
      auto number = column.getAsInteger();
      if (!number || *number <= 0) {
        emitError(location) << "tuning profile '" << family << "' values must be positive integers";
        return failure();
      }
      row.push_back(*number);
    }
    if (llvm::is_contained(table, row)) {
      emitError(location) << "tuning profile '" << family << "' contains duplicate rows";
      return failure();
    }
    table.push_back(std::move(row));
  }
  return table;
}

} // namespace

FailureOr<TuningProfiles> TuningProfiles::read(
    Location location, ArrayRef<TuningProfileSource> defaults,
    StringRef overrideFilename) {
  TuningProfiles profiles;
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
    auto columns = root->getArray("columns");
    if (columns->size() != source.columns.size()) {
      emitError(location) << "tuning column schema disagrees for " << source.name;
      return failure();
    }
    for (size_t index = 0; index < source.columns.size(); ++index)
      if ((*columns)[index].getAsString() != source.columns[index]) {
        emitError(location) << "tuning column name/order disagrees for " << source.name;
        return failure();
      }
    Namespace space{source.columns.size(), {}};
    auto families = root->getObject("families");
    if (families->empty()) {
      emitError(location) << "default tuning namespace is empty: " << source.name;
      return failure();
    }
    for (const auto &entry : *families) {
      auto table = readTable(location, entry.second, space.width, entry.first);
      if (failed(table))
        return failure();
      space.families.try_emplace(entry.first, std::move(*table));
    }
    profiles.spaces.try_emplace(source.name, std::move(space));
  }
  if (overrideFilename.empty())
    return profiles;
  auto parsed = readJSON(location, overrideFilename);
  if (failed(parsed))
    return failure();
  auto root = parsed->getAsObject();
  if (!root) {
    emitError(location) << "tuning override requires a namespace object";
    return failure();
  }
  for (const auto &entry : *root) {
    auto found = profiles.spaces.find(entry.first);
    auto families = entry.second.getAsObject();
    if (found == profiles.spaces.end() || !families) {
      emitError(location) << "unknown or malformed tuning namespace '" << StringRef(entry.first) << "'";
      return failure();
    }
    Namespace &space = found->second;
    for (const auto &family : *families) {
      auto existing = space.families.find(family.first);
      if (existing == space.families.end()) {
        emitError(location) << "unknown tuning family '" << StringRef(entry.first) << "." << StringRef(family.first) << "'";
        return failure();
      }
      auto table = readTable(location, family.second, space.width, family.first);
      if (failed(table))
        return failure();
      existing->second = std::move(*table);
    }
  }
  return profiles;
}

FailureOr<ArrayRef<TuningProfiles::Row>>
TuningProfiles::get(StringRef name, StringRef family, Location location) const {
  auto space = spaces.find(name);
  if (space != spaces.end()) {
    auto table = space->second.families.find(family);
    if (table != space->second.families.end())
      return ArrayRef<Row>(table->second);
  }
  emitError(location) << "missing declared tuning family '" << name << "." << family << "'";
  return failure();
}

} // namespace intent::gpu
