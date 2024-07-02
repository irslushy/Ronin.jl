window.BENCHMARK_DATA = {
  "lastUpdate": 1719953911987,
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
          "id": "db1177491154b651b8f7b22ef0840ea7229ab88d",
          "message": "Merge pull request #19 from irslushy/develop-threading\n\nRefactored feature calculation code to improve performance, also improved testing suite.",
          "timestamp": "2024-07-02T14:51:27-06:00",
          "tree_id": "2064f6d5c7990431ca8c1984d1e5a3e4b0bace2f",
          "url": "https://github.com/irslushy/Ronin.jl/commit/db1177491154b651b8f7b22ef0840ea7229ab88d"
        },
        "date": 1719953910748,
        "tool": "julia",
        "benches": [
          {
            "name": "features/10",
            "value": 864330994.5,
            "unit": "ns",
            "extra": "gctime=14796590.5\nmemory=555150920\nallocs=7063838\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      }
    ]
  }
}