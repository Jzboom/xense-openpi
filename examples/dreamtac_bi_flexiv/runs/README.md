# Dream-Tac run presets

Run YAML keys are the fields of `main.Args`. Select one with
`--args.run <name>`; command-line flags override values from the YAML.

The supplied `dry-run.yaml` intentionally does not select a robot recipe. The
operator must still pass `--args.robot-recipe <name-or-path>` so a copied launch
command cannot silently connect to the wrong physical bench.
