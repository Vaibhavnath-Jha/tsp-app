import json
import requests

import numpy as np
import pandas as pd

import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import PolyLineTextPath
from folium.plugins import AntPath

from src.tsp import tspModel


st.set_page_config(layout="wide")

with open("available_countries.json", "r") as fp:
    LIST_OF_COUNTRIES = json.load(fp)

SUPPORTED_FORMULATIONS = {
    "Miller-Tucker-Zemlin": "MTZ",
    "Explicit-DFJ": "EDFJ",
    "Iterative-DFJ": "IDFJ",
}


@st.dialog(title="Info")
def formulation_info(selected):
    if selected == "Miller-Tucker-Zemlin":
        st.write(
            "The MTZ *(Miller-Tucker-Zemlin)* formulation eliminates subtours by introducing auxiliary variables that represent the order in which each city is visited in the tour. These variables enable linear constraints to enforce a valid tour without subtours."
        )
    elif selected == "Explicit-DFJ":
        st.write(
            "The Explicit-DFJ *(Dantzig-Fulkerson-Johnson)* formulation for the TSP adds a subtour elimination constraint for every nontrivial subset of cities—that is, it considers the entire powerset of the city set (except the empty and full sets) to prevent subtours. This approach becomes computationally infeasible when the number of cities exceeds about 10, due to the exponential growth in the number of subsets."
        )
    elif selected == "Iterative-DFJ":
        st.write(
            "The Iterative-DFJ *(Dantzig-Fulkerson-Johnson)* method repeatedly solves the TSP model, adding subtour elimination constraints whenever subtours are found in the current solution, and continues this process until the final solution forms a valid tour visiting all cities exactly once."
        )


DEFAULTS = {
    "solutionFound": False,
    "obj_val": 0,
    "path": [],
    "stats": {},
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


def reset_to_default_view():
    """Resets the session state to its default values."""
    for key, val in DEFAULTS.items():
        st.session_state[key] = val
    st.info("Inputs changed. Click 'Run Optimization' to generate a new solution.")


@st.cache_data(show_spinner=False)
def fetchCities(country):
    base_url = "https://datasets-server.huggingface.co/filter"
    params = {
        "dataset": "jamescalam/world-cities-geo",
        "config": "default",
        "split": "train",
        "where": f"country='{country}'",
        "length": 100,
    }

    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        st.error("Something went wrong while fetching the cities.")

    url = f"https://services9.arcgis.com/l9yXFvhjz46ekkZV/arcgis/rest/services/Countries_Centroids/FeatureServer/0/query?where=COUNTRY+%3D+%27{country}%27&outFields=*&f=pgeojson"

    country_center = requests.get(url)
    if country_center.status_code != 200:
        st.error(
            "Something went wrong while fetching coordinates for the selected country."
        )

    country_centroid = country_center.json()["features"][0]["properties"]

    return response.json()["rows"], [
        country_centroid["latitude"],
        country_centroid["longitude"],
    ]


def solve_tsp(city_df, distance_df, **config):
    obj_val = 0
    path = []
    stats = pd.DataFrame(columns=["Value"])

    try:
        sol_list, model = tspModel(city_df, distance_df, **config)
        sol, time_taken = sol_list
    except Exception as e:
        raise Exception(e)

    if model.solve_status.value not in [1, 2, 3, 5, 8]:
        st.error(
            f"Solution not found. Solver returned with status: {model.solve_status}"
        )
        st.session_state["solutionFound"] = False
    else:
        st.session_state["solutionFound"] = True
        stats_data = {
            "statistics": {
                "Objective Value": model.objective_value,
                "Solve Time (s)": model.solve_model_time
                if time_taken == 0
                else time_taken,
                "Status": model.status.name,
                "#Variables": model.num_variables,
                "#Constraints": model.num_equations,
            }
        }
        stats = pd.DataFrame.from_dict(
            stats_data["statistics"], orient="index", columns=["Value"]
        )
        stats["Value"] = stats["Value"].astype(str)

        path = [start := sol.n1.iloc[0]]
        while (next := sol.loc[sol.n1 == path[-1], "n2"].values[0]) != start:
            path.append(next)
        path.append(start)
        obj_val = model.objective_value

    st.session_state["obj_val"] = obj_val
    st.session_state["path"] = path
    st.session_state["stats"] = stats


def prepInput():
    country_centroid = [51.1657, 10.4515]  # DE
    city_df = pd.DataFrame(columns=["row.city", "row.latitude", "row.longitude"])

    with st.sidebar:
        selected_country = st.selectbox(
            "Choose a country",
            LIST_OF_COUNTRIES,
            index=None,
            placeholder="Type or select a country..",
            help="This tries to fetch 100 cities with their geolocation data from a Kaggle dataset",
            on_change=reset_to_default_view,
        )
        st.write("")
        st.write("")
        st.write("")
        select_formulation = st.radio(
            "Choose a formulation for TSP:",
            SUPPORTED_FORMULATIONS.keys(),
            index=0,
            help="Select the type of formulation used to eliminate subtours.",
            on_change=reset_to_default_view,
        )

        if st.button("Info", type="secondary"):
            formulation_info(select_formulation)

        st.divider()

        solver = st.text_input("Solver", value="CPLEX", on_change=reset_to_default_view)
        number_of_nodes = st.number_input(
            "Maximum nodes to consider for TSP",
            min_value=1,
            max_value=100,
            value=10,
            on_change=reset_to_default_view,
        )
        seed = st.number_input(
            "Random Seed for selecting the starting node",
            min_value=1,
            max_value=100,
            value=49,
            on_change=reset_to_default_view,
        )
        timeLimit = st.number_input(
            "Time limit (seconds)",
            min_value=1,
            max_value=60,
            value=30,
            on_change=reset_to_default_view,
        )

        if selected_country:
            with st.spinner("Fetching Cities..."):
                city_data, country_centroid = fetchCities(selected_country)
                city_df = pd.json_normalize(city_data)
                city_df = city_df[["row.city", "row.latitude", "row.longitude"]]
                try:
                    city_df = city_df.sample(
                        number_of_nodes, random_state=seed
                    ).reset_index(drop=True)
                except ValueError:
                    raise Exception(
                        "Maxnimum nodes is greater than number of cities available. Reduce maximum number of nodes."
                    )

    input_config = {
        "time_limit": timeLimit,
        "solver": solver,
        "maxnodes": number_of_nodes,
        "formulation": SUPPORTED_FORMULATIONS[select_formulation],
        "centroid": country_centroid,
        "country": selected_country,
    }

    return city_df, input_config


@st.cache_data(show_spinner=False)
def euclidean_distance_matrix(coords):
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    return dist_matrix


@st.cache_data
def plot_solution(city_df: pd.DataFrame, centroid: list[float], path: list):
    country_map = folium.Map(location=centroid, zoom_start=5)

    if len(city_df):
        for idx, (name, lat, long) in enumerate(
            city_df.itertuples(index=False, name=None)
        ):
            if idx == 0:
                folium.Marker(
                    [lat, long],
                    popup=name,
                    icon=folium.Icon(color="green", prefix="fa", icon="location-dot"),
                ).add_to(country_map)
            else:
                # Standard marker for others
                folium.Marker(
                    [lat, long], popup=name, icon=folium.Icon(icon="info-sign")
                ).add_to(country_map)

    if len(path):
        path_df = pd.DataFrame({"row.city": path})
        final_path = path_df.merge(city_df, on="row.city", how="left")

        AntPath(
            locations=final_path[["row.latitude", "row.longitude"]],
            color="#FF0000",
            delay=1000,
            dash_array=[20, 30],
            weight=5,
            opacity=0.8,
        ).add_to(country_map)

        final_path = pd.concat([final_path, final_path.iloc[[0]]], ignore_index=True)

        polyline = folium.PolyLine(
            final_path[["row.latitude", "row.longitude"]],
        ).add_to(country_map)

        arrows = PolyLineTextPath(
            polyline,
            "➔",
            repeat=True,
            offset=7,
            attributes={"fill": "red", "font-weight": "light", "font-size": "20"},
        )
        arrows.add_to(country_map)

    return country_map


def main():
    st.title("GAMSPy TSP Solver")

    city_df, config = prepInput()

    with st.sidebar:
        st.divider()
        st.write("")
        st.write("")
        run_opt = st.button("Run Optimization", type="primary")

    maxnodes = config["maxnodes"]
    formulation = config["formulation"]
    country_centroid = config["centroid"]

    if formulation == "EDFJ" and maxnodes > 10:
        st.toast(
            f"Explicit-DFJ would generate {2**maxnodes} subsets which might not work with a Demo License.",
            icon="⚠️",
        )

    col1, col2 = st.columns(2)

    with col2:
        output_placeholder = st.empty()
        if run_opt:
            dist_mat = euclidean_distance_matrix(
                city_df[["row.latitude", "row.longitude"]].to_numpy()
            )
            dist_df = pd.DataFrame(
                dist_mat, index=city_df["row.city"], columns=city_df["row.city"]
            )
            distance_df = dist_df.reset_index().melt(
                id_vars="row.city", var_name="to_city", value_name="distance"
            )
            named_formulation = [k for k, v in SUPPORTED_FORMULATIONS.items() if v == formulation]
            with st.spinner(f"Solving TSP with {named_formulation[0]} formulation..."):
                output_placeholder.empty()
                if city_df.empty:
                    raise st.error("Select a country first")
                print(f"Solving TSP-{formulation} for {config["country"]}...")
                solve_tsp(city_df, distance_df, **config)


        if st.session_state["solutionFound"]:
            with output_placeholder.container():
                st.success("Solution Found!")
                st.markdown("### Optimal Tour:")
                st.write(" ➡️ ".join(st.session_state["path"]))
                st.markdown("### Total distance travelled:")
                st.markdown(
                    f"{st.session_state['obj_val'] * 100: .2f} kms",
                )
                st.markdown("### Model Statistics:")
                st.dataframe(st.session_state["stats"])
        else:
            output_placeholder.empty()

    with col1:
        path = st.session_state.get("path", [])
        country_map = plot_solution(city_df, country_centroid, path=path)
        with st.spinner("Plotting solution path..."):
            st_folium(country_map, use_container_width=True, height=700)


if __name__ == "__main__":
    main()
