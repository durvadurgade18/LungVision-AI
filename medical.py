import math
import requests


# ===============================
# Distance Calculation
# ===============================

def calculate_distance(lat1, lon1, lat2, lon2):

    R = 6371  # Earth radius in KM

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return round(R * c, 2)


# ===============================
# Hospital Search
# ===============================

def get_nearby_hospitals(latitude, longitude):

    query = f"""
    [out:json][timeout:25];

    (
      nwr["amenity"="hospital"](around:15000,{latitude},{longitude});
      nwr["healthcare"="hospital"](around:15000,{latitude},{longitude});
      nwr["amenity"="clinic"](around:15000,{latitude},{longitude});
    );

    out center;
    """

    servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter"
    ]


    data = None


    for server in servers:

        try:

            print("Trying server:", server)

            response = requests.post(
                server,
                data=query,
                headers={
                    "User-Agent": "LungVision-AI"
                },
                timeout=30
            )


            print(
                "Status:",
                response.status_code
            )


            if response.status_code == 200:

                data = response.json()
                break


        except Exception as e:

            print(
                "Overpass Error:",
                e
            )


    if data is None:

        print("No Overpass data")

        return []



    print(
        "Places found:",
        len(data.get("elements", []))
    )



    hospitals = []

    added = set()



    for item in data.get("elements", []):


        tags = item.get(
            "tags",
            {}
        )


        name = tags.get(
            "name"
        )


        if not name:
            continue



        clean_name = name.lower().strip()


        if clean_name in added:
            continue


        added.add(clean_name)



        if item["type"] == "node":

            lat = item.get("lat")
            lon = item.get("lon")


        else:

            center = item.get(
                "center",
                {}
            )

            lat = center.get("lat")
            lon = center.get("lon")



        if lat and lon:

            hospitals.append({

                "name": name,

                "distance": calculate_distance(
                    latitude,
                    longitude,
                    lat,
                    lon
                ),

                "maps":
                f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"

            })



    hospitals.sort(
        key=lambda x:x["distance"]
    )


    return hospitals[:10]