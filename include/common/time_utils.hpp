#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

class TimeInterval {

private:
  std::chrono::steady_clock::time_point start_;
  std::chrono::steady_clock::time_point end_;
  bool start_set_ = false;
  bool end_set_ = false;

public:
  void RecordStart();
  void RecordEnd();
  std::chrono::duration<double> StartTime() const;
  std::chrono::duration<double> EndTime() const;
  std::chrono::duration<double> ElapsedTime() const;
};

class TimeIntervals {

public:
  void AttachIntervals(std::vector<std::string> names);
  void RecordStart(std::string name);
  void RecordEnd(std::string name);
  std::chrono::duration<double> StartTime(std::string name) const;
  std::chrono::duration<double> EndTime(std::string name) const;
  std::chrono::duration<double> ElapsedTime(std::string name) const;

private:
  std::unordered_map<std::string, TimeInterval> data_;
};