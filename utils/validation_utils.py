def validate_alphanumeric(text):
    """Validate text to only allow alphanumeric characters"""
    return text.isalnum() or text == ''

def validate_alphanumeric_with_spaces(text):
    """Validate text to allow alphanumeric characters and spaces"""
    return all(c.isalnum() or c.isspace() for c in text) or text == ''

def validate_numeric(text):
    """Validate text to only allow numbers"""
    return text.isdigit() or text == ''