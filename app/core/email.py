class EmailSendFailed(Exception):
    """SMTP refused or failed; routes turn this into 502."""
