from datetime import datetime, timedelta
from collections import defaultdict
from notifications.schedule_configurations import *
from abc import ABC, abstractmethod
from collections import defaultdict

class ScheduleCalculatorBase(ABC):
    """
    Abstract base class for schedule calculation strategies.
    """
    @abstractmethod
    def calculate_schedule_times(self, all_logs):
        """
        Return a list of scheduled times as strings "HH:MM".
        """
        pass

    # Utility methods
    def _bucket_logs(self, logs):
        bucket_energy = defaultdict(int)
        for t, kcal in logs:
            minutes = self._time_to_minutes(t)
            bucket = (minutes // BUCKET_MINUTE_SIZE) * BUCKET_MINUTE_SIZE
            bucket_energy[bucket] += kcal
        return bucket_energy

    @staticmethod
    def _time_to_minutes(t):
        h, m = map(int, t.split(":"))
        return h * 60 + m

    @staticmethod
    def _minutes_to_time(m):
        h = (m // 60) % 24
        min_rem = m % 60
        return f"{h:02d}:{min_rem:02d}"


class GreedyScheduleCalculator(ScheduleCalculatorBase):
    """
    Original greedy algorithm with non-maximum suppression (NMS).
    Picks the top energy buckets first, respecting MIN_MINUTES_APART.
    """

    def calculate_schedule_times(self, all_logs):
        if not all_logs:
            return []

        bucket_energy = self._bucket_logs(all_logs)
        sorted_buckets = sorted(bucket_energy.items(), key=lambda x: -x[1])

        chosen = []
        for minutes, _ in sorted_buckets:
            if all(abs(minutes - c) >= MIN_MINUTES_APART for c in chosen):
                chosen.append(minutes)
            if len(chosen) == MAX_NOTIFICATIONS:
                break

        chosen.sort()
        return [self._minutes_to_time(m + NOTIFICATION_MINUTE_BUFFER) for m in chosen]


class SegmentedScheduleCalculator(ScheduleCalculatorBase):
    """
    Improved algorithm: splits the day into segments and picks top bucket per segment,
    ensuring notifications are spread across the day.
    """

    def calculate_schedule_times(self, all_logs):
        if not all_logs:
            return []

        bucket_energy = self._bucket_logs(all_logs)
        scheduled_minutes = self._select_schedule(bucket_energy)
        return [self._minutes_to_time(m + NOTIFICATION_MINUTE_BUFFER) for m in scheduled_minutes]

    def _select_schedule(self, bucket_energy):
        # Divide day into segments:
        # 0-4am, 4-11am, 11am-3pm, 3pm-5pm, 5pm-
        # segments = [(0, 4 * 60), (4*60, 11*60), (11+60, 15*60), (15 * 60, 17 * 60), (17*60, 24*60)]
        segments = [(0, 11 * 60), (11+60, 16*60), (16 * 60, 24 * 60)]

        chosen = []

        for start, end in segments:
            segment_buckets = [(m, kcal) for m, kcal in bucket_energy.items() if start <= m < end]
            if not segment_buckets:
                continue
            segment_buckets.sort(key=lambda x: -x[1])
            for m, _ in segment_buckets:
                if all(abs(m - c) >= MIN_MINUTES_APART for c in chosen):
                    chosen.append(m)
                    break

        # Fill remaining notifications if fewer than MAX_NOTIFICATIONS
        if len(chosen) < MAX_NOTIFICATIONS:
            remaining = sorted(bucket_energy.items(), key=lambda x: -x[1])
            for m, _ in remaining:
                if m not in chosen and all(abs(m - c) >= MIN_MINUTES_APART for c in chosen):
                    chosen.append(m)
                    if len(chosen) == MAX_NOTIFICATIONS:
                        break

        chosen.sort()
        return chosen


def weightedAvgSegmented(base_segments, logs):
    segments = [[] for start, end in base_segments]
    for (t, energy) in logs:
        minutes = ScheduleCalculatorBase._time_to_minutes(t)
        for i, (start, end) in enumerate(base_segments):
            if minutes >= start and minutes < end:
                segments[i].append((minutes, energy))

    for segment in segments:
        totalEnergy = sum([energy for _, energy in segment])
        time = 0
        for minutes, energy in segment:
            weight = energy / totalEnergy
            time += weight * minutes
        print("-" * 50)
        print([f"{(min/60):.1f}" for min, eng in segment])
        print(time, time/60)


def tester():
    nums = [8, 5, 2, 3, 4, 5, 2, 12, 32, 1, 1, 2]
    arr =  [7, 5, 5, 5, 2, 3, 4, 14, 12, 1, 2, 3]

    s = sum(nums)
    t = 0
    for i, n in enumerate(nums):
        weight = n / s
        t += weight * arr[i]
    print(t)
    print(sum(arr) / len(nums))



if __name__ == "__main__":
    tester()
    base_segments = [(0, 11 * 60), (11*60, 16*60), (16 * 60, 24 * 60)]
    logs = [
        ("08:05", 200), ("09:10", 150), ("10:25", 100),
        ("12:05", 250), ("14:15", 150), ("16:05", 100),
        ("18:10", 300), ("20:20", 200), ("22:15", 100),
        # Day 2
        ("08:30", 180), ("09:05", 130), ("10:45", 120),
        ("12:10", 260), ("14:05", 160), ("16:20", 90),
        ("18:30", 280), ("20:10", 220), ("22:00", 110),
        # Day 3
        ("08:15", 190), ("09:00", 140), ("10:50", 110),
        ("12:05", 240), ("14:20", 130), ("16:10", 120),
        ("18:05", 290), ("20:25", 210), ("22:10", 105),
        # Day 4
        ("08:00", 200), ("09:20", 150), ("10:40", 130),
        ("12:00", 250), ("14:10", 150), ("16:15", 100),
        ("18:15", 310), ("20:05", 200), ("22:20", 100),
        # Day 5
        ("08:10", 210), ("09:05", 140), ("10:35", 120),
        ("12:10", 260), ("14:15", 160), ("16:05", 90),
        ("18:20", 300), ("20:15", 220), ("22:10", 110)
    ]
    # weightedAvgSegmented(base_segments, logs)
