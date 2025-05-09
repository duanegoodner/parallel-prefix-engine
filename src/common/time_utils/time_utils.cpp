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

void TimeIntervals::AttachIntervals(std::vector<std::string> names) {
  for (const auto &name : names) {
    if (data_.count(name)) {
      throw std::runtime_error("Duplicate interval name: " + name);
    }
    data_[name] = TimeInterval();
  }
}

void TimeIntervals::RecordStart(std::string name) {
  data_.at(name).RecordStart();
}

void TimeIntervals::RecordEnd(std::string name) { data_.at(name).RecordEnd(); }

std::chrono::duration<double> TimeIntervals::StartTime(std::string name
) const {
  return data_.at(name).StartTime();
}

std::chrono::duration<double> TimeIntervals::EndTime(std::string name) const {
  return data_.at(name).EndTime();
}

std::chrono::duration<double> TimeIntervals::ElapsedTime(std::string name
) const {
  return data_.at(name).ElapsedTime();
}