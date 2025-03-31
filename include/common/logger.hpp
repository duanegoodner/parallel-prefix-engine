// ----------------------------------------------------------------------------
// logger.hpp
//
// Minimal logging utility for optional verbose output. Currently unused but
// available for future debug or profiling expansion.
// ----------------------------------------------------------------------------

#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

enum class LogLevel { INFO, DEBUG, ERROR };

class Logger {
public:
  static void SetVerbose(bool enabled) { verbose_enabled_ = enabled; }

  static void Log(LogLevel level, const std::string &message) {
    if (level == LogLevel::DEBUG && !verbose_enabled_)
      return;

    std::ostream &out = (level == LogLevel::ERROR) ? std::cerr : std::cout;
    out << "[" << ToString(level) << "] " << message << "\n";
  }

private:
  static inline bool verbose_enabled_ = false;

  static std::string ToString(LogLevel level) {
    switch (level) {
    case LogLevel::INFO:
      return "INFO";
    case LogLevel::DEBUG:
      return "DEBUG";
    case LogLevel::ERROR:
      return "ERROR";
    default:
      return "UNKNOWN";
    }
  }
};
