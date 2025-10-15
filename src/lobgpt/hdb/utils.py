from datetime import datetime, timedelta


def get_days(start_date: datetime, end_date: datetime) -> list[str]:
    """
    Generate a list of days between them as strings in 'YYYY-MM-DD' format.
    """
    days = []
    current_date = start_date

    while current_date <= end_date:
        days.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    return sorted(days)