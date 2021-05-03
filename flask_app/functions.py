def validate_file(filename, allowed_files):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_files

def is_authenticated(key):
    return key == 'atomas'