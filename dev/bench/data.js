window.BENCHMARK_DATA = {
  "lastUpdate": 1742836877734,
  "repoUrl": "https://github.com/irslushy/Ronin.jl",
  "entries": {
    "Julia benchmark result": [
      {
        "commit": {
          "author": {
            "email": "85644232+irslushy@users.noreply.github.com",
            "name": "Isaac Schluesche",
            "username": "irslushy"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "abe3b463a0b608e2f3b912893a9273cebc1938ec",
          "message": "Merge pull request #29 from irslushy/backend_refactor\n\nFixed bugs in the gate misclassification characterization code and made some backend performance improvements.",
          "timestamp": "2025-03-07T14:04:45-07:00",
          "tree_id": "fa159e22654f85e65f83cc648f162be7ecf45c53",
          "url": "https://github.com/irslushy/Ronin.jl/commit/abe3b463a0b608e2f3b912893a9273cebc1938ec"
        },
        "date": 1741381950921,
        "tool": "julia",
        "benches": [
          {
            "name": "features/10",
            "value": 857415697.5,
            "unit": "ns",
            "extra": "gctime=15033876.5\nmemory=555965128\nallocs=7069259\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "irschluesche@gmail.com",
            "name": "Isaac Schluesche",
            "username": "irslushy"
          },
          "committer": {
            "email": "irschluesche@gmail.com",
            "name": "Isaac Schluesche",
            "username": "irslushy"
          },
          "distinct": true,
          "id": "ff36fc4245e50040d1b5bd96acfd9ccd400c47d0",
          "message": "Fixed bug with multi-pass models that resulted in program crash upon no gates being passed to second sweep. Also removed extraneous print statements and reduced output from opening NCDatasets",
          "timestamp": "2025-03-24T11:13:32-06:00",
          "tree_id": "9e20f4f2a5d982c1be7a559b748ab048d82796ce",
          "url": "https://github.com/irslushy/Ronin.jl/commit/ff36fc4245e50040d1b5bd96acfd9ccd400c47d0"
        },
        "date": 1742836877070,
        "tool": "julia",
        "benches": [
          {
            "name": "features/10",
            "value": 854976345.5,
            "unit": "ns",
            "extra": "gctime=16160986.5\nmemory=555965080\nallocs=7069258\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      }
    ]
  }
}