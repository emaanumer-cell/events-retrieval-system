from pathlib import Path

import openpyxl

from src.models import Event, Parameter


def parse_events(filepath: str | Path) -> list[Event]:
    """Parse a tracking plan xlsx into consolidated Event objects.

    The xlsx has one row per parameter, so a single event spans multiple rows.
    A row with a non-empty Event Name starts a new event; rows with an empty
    Event Name are continuation rows that add parameters to the current event.

    Column structure:
    # | Event Name | Event Definition | Screen Name | Parameters | Parameter Descriptions | Sample Values | Key Event | Detailed Event Definition
    0 |     1      |        2          |      3      |     4      |          5             |       6       |     7     |           8
    """
    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    ws = wb.active

    events: list[Event] = []
    current_event: Event | None = None

    for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        event_name = row[1]
        event_definition = row[2]
        screen_name = row[3]
        param_name = row[4]
        param_description = row[5]
        sample_values = row[6]
        key_event = row[7] if len(row) > 7 else None
        detailed_event_definition = row[8] if len(row) > 8 else None

        # New event starts when Event Name column is populated
        if event_name is not None and str(event_name).strip():
            # Finalize previous event
            if current_event is not None:
                events.append(current_event)

            current_event = Event(
                event_name=str(event_name).strip(),
                event_definition=str(event_definition).strip() if event_definition else "",
                screen_name=str(screen_name).strip() if screen_name else "",
                key_event=str(key_event).strip() if key_event else "No",
                detailed_event_definition=str(detailed_event_definition).strip() if detailed_event_definition else "",
            )

        # Add parameter if present (applies to both new-event rows and continuation rows)
        if current_event is not None and param_name is not None and str(param_name).strip():
            current_event.parameters.append(
                Parameter(
                    name=str(param_name).strip(),
                    description=str(param_description).strip() if param_description else "",
                    sample_values=str(sample_values).strip() if sample_values else "",
                )
            )

    # Don't forget the last event
    if current_event is not None:
        events.append(current_event)

    wb.close()
    return events
