

def clean_abstract(abstract):
    try:
        if abstract.lower() == "unknown":
            return None
        else:
            return abstract
    except AttributeError:
        return None


def clean_doi(doi):
    valid_chars = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ./-_:')
    cleaned_doi = ''

    if 'unknown' in doi.lower():
        return ""

    for char in doi:
        if char in valid_chars:
            cleaned_doi = cleaned_doi + char
        else:
            break
    return cleaned_doi


def has_abstract(abstract):
    if abstract:
        return True
    else:
        return False