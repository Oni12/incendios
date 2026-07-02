import os
import xml.etree.ElementTree as ET
import re

KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}
KML_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "Incendios_2024_2025.kml")


def parse_polygons() -> list[dict]:
    tree = ET.parse(KML_PATH)
    root = tree.getroot()

    placemarks = root.findall(".//kml:Placemark", KML_NS)
    if not placemarks:
        polygons = root.findall(".//kml:Polygon", KML_NS)
        if not polygons:
            raise ValueError("No se encontró ningún <Polygon> o <Placemark> en el KML")
        
        result: list[dict] = []
        for polygon in polygons:
            coords_elem = polygon.find(".//kml:coordinates", KML_NS)
            if coords_elem is None or not coords_elem.text:
                continue
            raw = re.sub(r'\s*,\s*', ',', coords_elem.text.strip())
            points: list[list[float]] = []
            for token in raw.split():
                parts = token.split(",")
                if len(parts) >= 2:
                    try:
                        lng = float(parts[0])
                        lat = float(parts[1])
                        points.append([lat, lng])
                    except ValueError:
                        continue
            if points:
                result.append({"coordinates": points, "year": None})
        return result

    result: list[dict] = []
    for placemark in placemarks:
        polygon = placemark.find(".//kml:Polygon", KML_NS)
        if polygon is None:
            continue

        coords_elem = polygon.find(".//kml:coordinates", KML_NS)
        if coords_elem is None or not coords_elem.text:
            continue

        # Normalizar coordenadas quitando espacios alrededor de las comas
        raw = re.sub(r'\s*,\s*', ',', coords_elem.text.strip())
        points: list[list[float]] = []
        for token in raw.split():
            parts = token.split(",")
            if len(parts) >= 2:
                try:
                    lng = float(parts[0])
                    lat = float(parts[1])
                    points.append([lat, lng])
                except ValueError:
                    continue

        if points:
            # Determinar el año del incendio a partir del nombre o styleUrl
            name_elem = placemark.find("kml:name", KML_NS)
            name_text = name_elem.text if name_elem is not None else ""
            style_elem = placemark.find("kml:styleUrl", KML_NS)
            style_text = style_elem.text if style_elem is not None else ""
            
            year = None
            if "2024" in name_text or "2024" in style_text:
                year = 2024
            elif "2025" in name_text or "2025" in style_text:
                year = 2025

            result.append({"coordinates": points, "year": year})

    if not result:
        raise ValueError("No se encontraron polígonos válidos en el KML")

    return result
