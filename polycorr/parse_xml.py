"""Collection of methods to process data"""

import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm

def parse_xml(filename, ubound, lbound):
    """Method to parse XML files from CT measurements"""

    root = ET.parse(filename).getroot()
    n_s = {'ss':"urn:schemas-microsoft-com:office:spreadsheet"}
    print("Parsing Points")
    # Workbook/Worksheet Points/Table/Row
    points = np.zeros((len(root[2][0][1:]), 3))
    to_drop = []
    for idx, row in enumerate(root[2][0][1:]):
        point = [
            float(data.text)
            for data in row.findall(
                './/{urn:schemas-microsoft-com:office:spreadsheet}Data', n_s
            )[1:]
        ]
        if point[2] >= lbound and point[2] <= ubound:
            points[idx] = point
        else:
            to_drop.append(idx)

    print("Parsing Segments")
    # Workbook/Worksheet Segments/Table/Row
    segments = []
    for row in tqdm(root[3][0][1:]):
        data = row.findall('.//{urn:schemas-microsoft-com:office:spreadsheet}Data', n_s)[-1]
        segment = np.array(data.text.split(','), dtype=int)
        if len(set(to_drop).intersection(set(segment))) == 0:
            segments.append(segment)

    return points, segments
