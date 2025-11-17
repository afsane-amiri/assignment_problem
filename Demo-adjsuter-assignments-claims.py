#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install folium ortools geopy pandas numpy ipywidgets


# In[2]:


import folium
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from ortools.linear_solver import pywraplp
from ipywidgets import interact, IntSlider, HTML
from IPython.display import display
import random
from folium.plugins import HeatMap

# --- Continental US states with abbreviations ---
us_states = [
    ("AL", 32.806671, -86.791130), ("AZ", 33.729759, -111.431221),
    ("AR", 34.969704, -92.373123), ("CA", 36.116203, -119.681564),
    ("CO", 39.059811, -105.311104), ("CT", 41.597782, -72.755371),
    ("DE", 39.318523, -75.507141), ("FL", 27.766279, -81.686783),
    ("GA", 33.040619, -83.643074), ("ID", 44.240459, -114.478828),
    ("IL", 40.349457, -88.986137), ("IN", 39.849426, -86.258278),
    ("IA", 42.011539, -93.210526), ("KS", 38.526600, -96.726486),
    ("KY", 37.668140, -84.670067), ("LA", 31.169546, -91.867805),
    ("ME", 44.693947, -69.381927), ("MD", 39.063946, -76.802101),
    ("MA", 42.230171, -71.530106), ("MI", 43.326618, -84.536095),
    ("MN", 45.694454, -93.900192), ("MS", 32.741646, -89.678696),
    ("MO", 38.456085, -92.288368), ("MT", 46.921925, -110.454353),
    ("NE", 41.125370, -98.268082), ("NV", 38.313515, -117.055374),
    ("NH", 43.452492, -71.563896), ("NJ", 40.298904, -74.521011),
    ("NM", 34.840515, -106.248482), ("NY", 42.165726, -74.948051),
    ("NC", 35.630066, -79.806419), ("ND", 47.528912, -99.784012),
    ("OH", 40.388783, -82.764915), ("OK", 35.565342, -96.928917),
    ("OR", 44.572021, -122.070938), ("PA", 40.590752, -77.209755),
    ("RI", 41.680893, -71.511780), ("SC", 33.856892, -80.945007),
    ("SD", 44.299782, -99.438828), ("TN", 35.747845, -86.692345),
    ("TX", 31.054487, -97.563461), ("UT", 40.150032, -111.862434),
    ("VT", 44.045876, -72.710686), ("VA", 37.769337, -78.169968),
    ("WA", 47.400902, -121.490494), ("WV", 38.491226, -80.954453),
    ("WI", 44.268543, -89.616508), ("WY", 42.755966, -107.302490)
]

def optimize_and_plot(storm_severity=5, num_adjusters=5, budget=2_000_000):
    np.random.seed(42)
    n_claims = storm_severity * 10
    capacity = max(int(np.ceil(n_claims / num_adjusters)), 1)

    # --- Claims ---
    claim_states = random.choices(us_states, k=n_claims)
    claims = pd.DataFrame({
        "id": range(n_claims),
        "abbr": [s[0] for s in claim_states],
        "lat": [s[1] for s in claim_states],
        "lon": [s[2] for s in claim_states]
    })

    # --- Adjusters ---
    adjuster_states = random.sample(us_states, num_adjusters)
    adjusters = pd.DataFrame({
        "id": range(num_adjusters),
        "abbr": [s[0] for s in adjuster_states],
        "lat": [s[1] for s in adjuster_states],
        "lon": [s[2] for s in adjuster_states]
    })

    # --- Cost matrix ---
    cost = np.zeros((n_claims, num_adjusters))
    for i in range(n_claims):
        for j in range(num_adjusters):
            cost[i,j] = geodesic((claims.lat[i], claims.lon[i]),
                                 (adjusters.lat[j], adjusters.lon[j])).km

    # --- GLOP solver ---
    solver = pywraplp.Solver.CreateSolver('GLOP')
    x = {(i,j): solver.NumVar(0,1,f"x[{i},{j}]")
         for i in range(n_claims) for j in range(num_adjusters)}

    for i in range(n_claims):
        solver.Add(sum(x[i,j] for j in range(num_adjusters)) == 1)
    for j in range(num_adjusters):
        solver.Add(sum(x[i,j] for i in range(n_claims)) <= capacity)
    solver.Add(sum(cost[i,j]*x[i,j] for i in range(n_claims) for j in range(num_adjusters)) <= budget)

    solver.Minimize(sum(cost[i,j]*x[i,j] for i in range(n_claims) for j in range(num_adjusters)))
    solver.Solve()

    # --- Assignments ---
    assignments = [(i,j,x[i,j].solution_value()*cost[i,j])
                   for i in range(n_claims) for j in range(num_adjusters)
                   if x[i,j].solution_value() > 1e-6]

    # --- KPIs ---
    total_cost = sum(c for _,_,c in assignments)
    avg_distance = np.mean([c for _,_,c in assignments])
    sla = np.mean([c <= 500 for _,_,c in assignments])
    utilization = np.mean([sum(x[i,j].solution_value() for i in range(n_claims))/capacity
                           for j in range(num_adjusters)])

    kpi_html = f"""
    <h4>📊 KPI Summary</h4>
    <ul>
      <li><b>Total Cost:</b> {total_cost:.1f} $</li>
      <li><b>Avg Distance:</b> {avg_distance:.1f} km</li>
      <li><b>SLA (&lt;500 km):</b> {sla*100:.1f}%</li>
      <li><b>Adjuster Utilization:</b> {utilization*100:.1f}%</li>
    </ul>
    """
    display(HTML(kpi_html))

    # --- Map ---
    m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="cartodbpositron")

    # --- Heatmap for claims (storm severity density) ---
    heat_data = [[row.lat, row.lon] for _, row in claims.iterrows()]
    HeatMap(heat_data, radius=25, blur=15, max_zoom=6).add_to(m)

    # Claims: pins
    for _, row in claims.iterrows():
        folium.Marker([row.lat, row.lon],
                      popup=f"Claim: {row.abbr}",
                      icon=folium.Icon(color="blue", icon="flag")).add_to(m)

    # Adjusters: person icons
    for _, row in adjusters.iterrows():
        folium.Marker([row.lat, row.lon],
                      popup=f"Adjuster: {row.abbr}",
                      icon=folium.Icon(color="red", icon="user")).add_to(m)

    # Assignments with SLA color + tooltip
    for (i,j,c) in assignments:
        color = "green" if c <= 500 else "red"
        folium.PolyLine([(claims.lat[i], claims.lon[i]),
                         (adjusters.lat[j], adjusters.lon[j])],
                        color=color, weight=max(1, 3*c/np.max(cost)),
                        opacity=0.7, tooltip=f"{c:.1f} km").add_to(m)

    # Legend
    legend_html = """
     <div style="position: fixed; 
                 bottom: 50px; left: 50px; width: 180px; height: 130px; 
                 border:2px solid grey; z-index:9999; font-size:14px;
                 background-color:white; padding: 10px;">
     <b>Legend</b><br>
     <span style="color:blue;">📌</span> Claim<br>
     <span style="color:red;">👤</span> Adjuster<br>
     <span style="color:green;">──</span> Within SLA<br>
     <span style="color:red;">──</span> SLA Violation<br>
     <span style="background:rgba(255,0,0,0.4);">■</span> Storm Intensity
     </div>
     """
    m.get_root().html.add_child(folium.Element(legend_html))

    display(m)

# --- Interactive sliders ---
interact(
    optimize_and_plot,
    storm_severity=IntSlider(value=5, min=1, max=10, step=1, description="Storm Severity"),
    num_adjusters=IntSlider(value=5, min=1, max=20, step=1, description="Adjusters"),
    budget=IntSlider(value=2_000_000, min=100_000, max=5_000_000, step=100_000, description="Budget")
)


# In[ ]:




