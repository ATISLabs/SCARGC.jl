using Pkg

Pkg.activate(".")

using Documenter, SCARGC

makedocs(
        format = Documenter.HTML(
            assets=[asset("https://fonts.googleapis.com/css?family=Montserrat|Source+Code+Pro&display=swap", class=:css)]
        ),
        sitename="SCARGC.jl",
        authors = "Gabriel Marinho",
        pages = [
            "Home" => "index.md",
            "Functions" => "functions.md"
        ]
    )