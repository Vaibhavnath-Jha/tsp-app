import json
import time

import gamspy as gp
from gamspy.exceptions import ValidationError, GamspyException

import networkx as nx
import numpy as np
import pandas as pd


def mtz_formulation(m: gp.Container) -> gp.Equation:
    n1, n2, i, j, ij, X, start_point = m.getSymbols(
        ["n1", "n2", "i", "j", "allowed_arcs", "X", "start_point"]
    )

    P = gp.Variable(m, "p", type="positive", domain=n1)

    P.fx[start_point] = 0
    P.lo[n1].where[~n1.sameAs(start_point)] = 1
    P.up[n1] = gp.Card(n2) - 1

    eq_mtz = gp.Equation(
        m,
        "eq_mtz",
        description="Miller, Tucker and Zemlin subtour elimination",
        domain=[n1, n2],
    )

    eq_mtz[i, j].where[ij[i, j] & ~i.sameAs(start_point) & ~j.sameAs(start_point)] = P[
        i
    ] - P[j] + 1 <= gp.Card(i) * (1 - X[i, j])

    return [eq_mtz]


def explicit_dfj_formulation(m: gp.Container) -> gp.Equation:
    n1, i, j, ij, X = m.getSymbols(["n1", "i", "j", "allowed_arcs", "X"])

    s = gp.Set(
        m,
        "s",
        domain="*",
        description="Powerset",
    )
    sn = gp.Set(m, "sn", domain=[s, n1], description="subset_membership")

    subset_size = gp.Parameter(
        m,
        "subset_size",
        domain=[s],
        description="Number of elements in each subset",
    )

    filtered_set = gp.Set(
        m,
        "filtered_powetset",
        domain=[s],
        description="Subsets excluding empty, singletons, and full set",
    )

    s.setRecords([f"s{subset}" for subset in range(2 ** len(i))])
    sn[s, n1].where[
        i[n1]
        & gp.math.mod(
            gp.math.floor((gp.Ord(s) - 1) / gp.math.power(2, gp.Ord(n1) - 1)), 2
        )
        == 1
    ] = True

    subset_size[s] = gp.Sum(i.where[sn[s, i]], gp.Number(1))
    filtered_set[s].where[(subset_size[s] > 1) & (subset_size[s] < len(i))] = True

    eq_dfj = gp.Equation(
        m,
        "eq_dfj",
        domain=[s],
        description="Explicit Dantzig-Fulkerson-Johnson subtour elimination constraint.",
    )
    eq_dfj[filtered_set[s]] = (
        gp.Sum(gp.Domain(i, j).where[(ij[i, j]) & (sn[s, i]) & (sn[s, j])], X[i, j])
        <= subset_size[s] - 1
    )

    return [eq_dfj]


def find_illegal_subtour(sol: pd.DataFrame, start_point):
    G = nx.DiGraph()
    G.add_edges_from(
        [(i, j) for i, j in sol[["n1", "n2"]].itertuples(index=False, name=None)]
    )
    components = list(nx.strongly_connected_components(G))

    return [list(comp) for comp in components if start_point not in comp]


def tspModel(
    nodes_recs, distance_recs, **options
) -> tuple[list[pd.DataFrame, float], gp.Model]:
    solver: str = options["solver"]
    time_limit: int = options["time_limit"]
    maxnodes: int = options["maxnodes"]
    formulation: str = options["formulation"]

    m = gp.Container()

    nodes = gp.Set(m, name="set_of_nodes", domain="*")

    start_point = gp.Set(
        m,
        "start_point",
        domain=nodes,
        is_singleton=True,
    )

    n1 = gp.Alias(m, name="n1", alias_with=nodes)
    n2 = gp.Alias(m, name="n2", alias_with=nodes)

    i = gp.Set(m, name="i", domain=nodes, description="subset of nodes")

    j = gp.Alias(m, name="j", alias_with=i)

    ij = gp.Set(m, name="allowed_arcs", domain=[n1, n2])

    distance = gp.Parameter(m, name="distance_matrix", domain=[n1, n2])

    total_cost = gp.Variable(m, name="total_cost")

    X = gp.Variable(
        m,
        name="x",
        type="binary",
        domain=[n1, n2],
        description="decision variable - leg of trip",
    )

    objective_function = gp.Equation(m, name="tsp_objective_function")
    objective_function[...] = gp.Sum(ij[i, j], distance[i, j] * X[i, j]) == total_cost

    eq_enter_once = gp.Equation(m, "eq_enter_once", domain=[n1])
    eq_enter_once[j] = (
        gp.Sum(i.where[ij[i, j]], X[i, j]) == 1
    )

    eq_leave_once = gp.Equation(m, "eq_leave_once", domain=[n1])
    eq_leave_once[i] = (
        gp.Sum(j.where[ij[i, j]], X[i, j]) == 1
    )

    nodes.setRecords(nodes_recs["row.city"])
    start_point.setRecords([nodes.records.iloc[0]["uni"]])
    distance.setRecords(distance_recs)

    i[n1].where[gp.Ord(n1) <= maxnodes] = True
    ij[i, j].where[~i.sameAs(j)] = True

    if formulation.upper() == "IDFJ":
        s = gp.Set(
            m,
            "s",
            domain="*",
            description="Powerset",
            records=range(1000),
        )

        active_cut = gp.Set(m, "active_cut", domain=[s])
        sn = gp.Set(m, "sn", domain=[s, n1], description="subset_membership")

        eq_dfj_iter = gp.Equation(m, "eq_dfj_iter", domain=[s])

        eq_dfj_iter[active_cut] = (
            gp.Sum(
                gp.Domain(i, j).where[
                    (ij[i, j]) & (sn[active_cut, i]) & (sn[active_cut, j])
                ],
                X[i, j],
            )
            <= gp.Sum(i.where[sn[active_cut, i]], gp.Number(1)) - 1
        )

        tsp = gp.Model(
            m,
            name="tsp",
            problem="MIP",
            sense=gp.Sense.MIN,
            objective=total_cost,
            equations=[objective_function, eq_enter_once, eq_leave_once, eq_dfj_iter],
        )

        tsp.solve(solver=solver, options=gp.Options(time_limit=time_limit))

        cnt = 0
        MAXCUTS = len(s)
        tot_time = 0
        set_of_tour = gp.Set(m, "set_of_tour", domain=[n1])

        while True:
            start = time.time()
            sol = X[...].where[X.l > 0.5].records
            subtours = find_illegal_subtour(
                sol, start_point=start_point.records["set_of_nodes"][0]
            )

            if not subtours:
                print("***All illegal subtours are removed. Soution found!***")
                break

            if cnt + len(subtours) > MAXCUTS:
                raise GamspyException(
                    f"Found {len(subtours)} illegal subtours, but adding them would"
                    f" exceed the cut limit of {MAXCUTS}."
                )

            for idx, tour in enumerate(subtours, start=cnt):
                set_of_tour.setRecords(tour)
                sn[idx, set_of_tour] = True

            cnt += len(subtours)
            print(
                f"illegal subtours in current solution: {len(subtours)} | total subtours: {cnt}"
            )
            active_cut[s] = gp.Ord(s) <= cnt
            tsp.solve(solver=solver, options=gp.Options(time_limit=time_limit))
            tot_time += time.time() - start

            if tot_time > time_limit:
                print("Total timelimit reached. Stopping!")
                break

        return [sol, tot_time], tsp

    elif formulation.upper() == "MTZ":
        subtour_eliminate_equation = mtz_formulation(m)
    elif formulation.upper() == "EDFJ":
        subtour_eliminate_equation = explicit_dfj_formulation(m)
    else:
        raise ValidationError(
            f"Wrong choice of formulation >{formulation}<. Choose from ['MTZ', 'EDFJ', 'IDFJ']"
        )

    tsp = gp.Model(
        m,
        name="tsp",
        problem="MIP",
        sense=gp.Sense.MIN,
        objective=total_cost,
        equations=[objective_function, eq_enter_once, eq_leave_once]
        + subtour_eliminate_equation,
    )

    tsp.solve(solver=solver, options=gp.Options(time_limit=time_limit))
    sol = X[...].where[X.l > 0.5].records

    return [sol, 0], tsp


def main():
    maxnodes = 10
    time_limit = 30
    solver = "CPLEX"
    formulation = "MTZ"
    options = {
        "solver": solver,
        "time_limit": time_limit,
        "maxnodes": maxnodes,
        "formulation": formulation,
    }

    def euclidean_distance_matrix(coords):
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        return dist_matrix

    with open(r"..\germany_cities.json", "r") as fp:
        city_data = json.load(fp)

    city_df = pd.json_normalize(city_data)
    dist_mat = euclidean_distance_matrix(
        city_df[["row.latitude", "row.longitude"]].to_numpy()
    )
    dist_df = pd.DataFrame(
        dist_mat, index=city_df["row.city"], columns=city_df["row.city"]
    )
    distance_df = dist_df.reset_index().melt(
        id_vars="row.city", var_name="to_city", value_name="distance"
    )

    tspModel(nodes_recs=city_df, distance_recs=distance_df, **options)


if __name__ == "__main__":
    main()
