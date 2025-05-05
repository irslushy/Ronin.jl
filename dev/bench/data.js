window.BENCHMARK_DATA = {
  "lastUpdate": 1746467391912,
  "repoUrl": "https://github.com/irslushy/Ronin.jl",
  "entries": {
    "Julia benchmark result": [
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
          "id": "9160d039bbb7e087ff650b674653b03ffb0340fd",
          "message": "Fixed bug where error thrown in calculate features would result in cfradial file not being properly closed",
          "timestamp": "2025-05-05T11:42:02-06:00",
          "tree_id": "b05a62eb9ed2ff1b2028086d20b8401d85e426b8",
          "url": "https://github.com/irslushy/Ronin.jl/commit/9160d039bbb7e087ff650b674653b03ffb0340fd"
        },
        "date": 1746467389589,
        "tool": "julia",
        "benches": [
          {
            "name": "features/10",
            "value": 857469606,
            "unit": "ns",
            "extra": "gctime=16159814.5\nmemory=555965128\nallocs=7069259\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      }
    ]
  }
}