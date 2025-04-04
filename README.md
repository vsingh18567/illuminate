# illuminate
Agents for data science - a work in progress.

## Usage

Install [`wkhtmltopdf`](https://wkhtmltopdf.org/) to enable PDF creation.

Install `illuminate` locally.

```bash
pip install -e .
```

Run the agent from the directory that has a prompt and the data you want to analyze (the `samples` directory has some examples).

```bash
cd samples/indian_traffic
illuminate
```

## Development
`uv` is used to manage the dependencies.

```bash
pip install uv
```

## Todos

- [ ] Figure out if there's a way for tools to return files.
- [ ] Add a database tool (probably just sqlite for now)
- [ ] Cleanup files that are created by the agents but aren't needed.
- [ ] WAY more testing
- [ ] The output is still extremely average. The goal is to have the agents produce work at a college-student level.
- [ ] Work on performance - the agents are slow. The best way to improve is to get the planning agent to plan larger (and fewer) steps.


