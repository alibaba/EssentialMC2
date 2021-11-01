// Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

#include <vector>
#include <set>
using namespace std;

// python thingy
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <pybind11/stl.h>


class UnionFind {
public:
    UnionFind(int N = 10) {
        uf.resize(N);
        cc_size.resize(N);
        for (int i = 0; i < N; ++ i) {
            uf[i] = i;
            cc_size[i] = 1;
        }
    }

    int find(int x) {
        return x == uf[x] ? x : uf[x] = find(uf[x]);
    }

    void union_(int p, int q) {
        int proot = find(p);
        int qroot = find(q);
        if (proot == qroot) {
            return ;
        } else if (cc_size[proot] > cc_size[qroot]) {
            cc_size[qroot] += cc_size[proot];
            uf[proot] = qroot;
        } else {
            cc_size[proot] += cc_size[qroot];
            uf[qroot] = proot;
        }
    }

    vector<int> uf, cc_size;
};

vector<int> calc_lcc(int ntrain, int nclass, vector<int> labels, vector<vector<int>> neighbors, vector<int> clean_idx) {
    vector<int> idx_of_largest_comps;

    vector<set<int>> neighbors_set(ntrain);
    for (int i = 0; i < ntrain; ++ i) if (clean_idx[i] == true) {
        for (int k: neighbors[i]) {
            if (labels[k] != labels[i] || clean_idx[k] == false) continue;
            neighbors_set[i].insert(k);
        }
    }

    for (int j = 0; j < nclass; ++ j) {
        auto UF = UnionFind(ntrain);

        for (int i = 0; i < ntrain; ++ i) if (labels[i] == j && clean_idx[i] == true) {
            for (int k: neighbors_set[i]) if (neighbors_set[k].find(i) != end(neighbors_set[k])) {
                UF.union_(i, k);
            }
        }

        int max_cc_idx = -1, max_cc_size = -1;
        for (int i = 0; i < ntrain; ++ i) {
            int rt = UF.find(i);
            if (rt == i) {
                if (max_cc_size < UF.cc_size[rt]) {
                    max_cc_size = UF.cc_size[rt];
                    max_cc_idx = rt;
                }
            }
        }

        for (int i = 0; i < ntrain; ++ i) {
            int fa = UF.find(i);
            if (fa == max_cc_idx)
                idx_of_largest_comps.push_back(i);
        }
    }

    return idx_of_largest_comps;
}

int main(void) {

    return 0;
}

PYBIND11_MODULE(CalcLargestConnectedComponentsIds, m) {
    m.doc() = "Calculate Largest Connected Components in graph, pybind11 interface";
    m.def("calc_lcc", &calc_lcc);
}

