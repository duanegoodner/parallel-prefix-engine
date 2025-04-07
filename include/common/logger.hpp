// ----------------------------------------------------------------------------
// logger.hpp
//
// Minimal logging utility for optional verbose output. Currently unused but
// available for future debug or profiling expansion.
// ----------------------------------------------------------------------------

#pragma once

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

enum class LogLevel { OFF, INFO, DEBUG, ERROR, COUNT };

namespace LogLevelUtils {
constexpr std::array<std::string_view, static_cast<size_t>(LogLevel::COUNT)>
    level_strings{"off", "info", "debug", "error"};

constexpr std::array<std::pair<std::string_view, LogLevel>, 4> string_to_level{
    {
        {"off", LogLevel::OFF},
        {"info", LogLevel::INFO},
        {"debug", LogLevel::DEBUG},
        {"error", LogLevel::ERROR},
    }};

constexpr std::string_view ToString(LogLevel level) {
  auto index = static_cast<size_t>(level);
  return index < level_strings.size() ? level_strings[index] : "UNKNOWN";
}

constexpr LogLevel FromString(std::string_view str) {
  for (const auto &[name, level] : string_to_level) {
    if (name == str)
      return level;
  }
  return LogLevel::OFF;
}
} // namespace LogLevelUtils

// Class Logger: Lightweight utility for conditional logging based on verbosity
// flag.
class Logger {
public:
  static void SetLogLevel(LogLevel level) { log_level_ = level; }

  static void Log(LogLevel level, const std::string &message) {
    if (log_level_ == LogLevel::OFF || level > log_level_)
      return;

    std::ostream &out = (level == LogLevel::ERROR) ? std::cerr : std::cout;
    out << "[" << LogLevelUtils::ToString(level) << "] " << message << "\n";
  }

private:
  static inline LogLevel log_level_ = LogLevel::OFF;
};