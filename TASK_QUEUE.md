# TASK QUEUE

## Task 001
Create BaseSource abstract class.

Requirements:
- File: src/data_sources/base_source.py
- Use abc.ABC
- Define:
    def fetch_historical(self, start_date: str, end_date: str)
- No implementation logic
- No logging
- No API calls
- No storage logic
- Only abc and typing imports

Also:
- Create minimal test in tests/test_base_source.py
- Ensure BaseSource cannot be instantiated

Stop after completion.
