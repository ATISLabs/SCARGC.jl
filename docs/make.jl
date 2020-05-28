using Pkg

Pkg.activate(".")

using Documenter, SCARGC
using DocumenterTools: Themes

Themes.compile(joinpath(@__DIR__,"src/assets/scargc-light.css"), joinpath(@__DIR__,"src/assets/themes/documenter-light.css"))

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


deploydocs(
    repo = "github.com/MarinhoGabriel/SCARGC.jl.git",
)