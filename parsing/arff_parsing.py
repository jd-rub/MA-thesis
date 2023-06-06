from .attribute_mapping import aarf_attribute_to_instrument_name

def parse_arff_line(line:str):
    if "@ATTRIBUTE" in line:
        attribute = line.split("'")[1]
        return True, attribute
    elif len(line) > 0 and line[0].isdigit():
        onset, tail = line.split(",", maxsplit=1)
        values = tail.split("]','[")
        values = [value.replace("'","") for value in values]
        values = [value.replace("[","") for value in values]
        values = [value.replace("]","") for value in values]
        values[-1] = values[-1].replace("\n", "")
        return False, onset, values
    else:
        return None

def parse_arff(filename):
    """Parses an arff annotation file from the tiny_aam dataset.

    Parameters
    ----------
    filename : str
        path to the .arff file

    Returns
    -------
    dict
        Dict of tuples {onset:[(instrument1, pitch1), instrument2, pitch2)]}
    """
    records = {}
    attributes = []
    lines = {}
    with open(filename, 'r') as fp:
        for line in fp.readlines():
            result = parse_arff_line(line)
            if result is not None:
                if result[0]:
                    # Attribute
                    attributes.append(result[1])
                else:
                    lines[result[1]] = result[2]

    for onset in lines:
        row = lines[onset]
        records[onset] = []
        for i, attribute in enumerate(attributes):
            if attribute == 'Onset time in seconds' or type(attribute) is not str:
                continue
            if row[i-1]:
                data = row[i-1]
                instrument = aarf_attribute_to_instrument_name[attribute]

                for pitch in data.split(","):
                    records[onset].append((instrument, pitch))
    return records