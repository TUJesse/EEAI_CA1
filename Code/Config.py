class Config:
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Type Columns to test
    TYPE_COLS = ['y2', 'y3', 'y4']
    CHAIN_TWO_TYPE_COLS = ['y2', 'y3']
    CLASS_COL = 'y2'
    GROUPED = 'y1'


    # Variables for CA 1 Task
    CLASS_COL = 'full_type'
    JOIN_CHAR = '^'
    EMPTY_TYPE = 'none'
    FORMATTED_TYPE_COLS = [f"formatted_{col}" for col in TYPE_COLS]
    CHAIN_TWO_COLS = [f"formatted_{col}" for col in CHAIN_TWO_TYPE_COLS]
    PRED_TYPE_COLS = [f"pred_{col}" for col in TYPE_COLS]
    OUTPUT_FILE = 'Output.txt'
