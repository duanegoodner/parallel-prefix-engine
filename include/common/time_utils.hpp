#pragma once

#include <chrono>

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