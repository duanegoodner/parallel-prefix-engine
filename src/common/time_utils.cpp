#include "common/time_utils.hpp"

#include <chrono>
#include <stdexcept>

void TimeInterval::RecordStart() {
  if (start_set_) {
    throw std::logic_error("Start time has already been recorded.");
  }
  start_ = std::chrono::steady_clock::now();
  start_set_ = true;
}

void TimeInterval::RecordEnd() {
  if (end_set_) {
    throw std::logic_error("End time has already been recorded.");
  }
  end_ = std::chrono::steady_clock::now();
  end_set_ = true;
}

std::chrono::duration<double> TimeInterval::StartTime() const {
  if (!start_set_) {
    throw std::logic_error("Start time has not been recorded.");
  }
  return std::chrono::duration<double>(start_.time_since_epoch());
}

std::chrono::duration<double> TimeInterval::EndTime() const {
  if (!end_set_) {
    throw std::logic_error("End time has not been recorded.");
  }
  return std::chrono::duration<double>(end_.time_since_epoch());
}

std::chrono::duration<double> TimeInterval::ElapsedTime() const {
  if (!start_set_ || !end_set_) {
    throw std::logic_error("Start and/or end time has not been recorded.");
  }
  return end_ - start_;
}