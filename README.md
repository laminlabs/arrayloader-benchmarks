To prevent dependency conflicts within benchmarks, we will use `hatch` to manage the environments around the `hydra` scripts.

To install hatch, see [their documentation](https://hatch.pypa.io/1.9/install/).

Then, to run a script, add it to the `pyproject.toml` `scripts` section with a given name i.e.,

```toml
[project.scripts]
my_pyproject_script = "arrayloader_benchmarks:my_click_script"
```

and then add that script under the desired environment in the `hatch.toml` file:

```toml
[envs.my_env_anme]
features = [ "my_env_defined_in_pyproject.toml" ]
dependncies = ["my_overriding_dependency"]  # TODO: this will literally override the features as opposed to installing the features and constraining. We should figure out what the hatch thing is to use enviornment variables, and then use `UV_CONSTRAINT`.
scripts.my_hatch_script_name = "my_pyproject_script {args}"
```

You can then run:

```bash
hatch run my_env_name:my_hatch_script_name args
```

in a dedicated environment.
