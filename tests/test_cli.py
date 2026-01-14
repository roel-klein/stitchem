from click.testing import CliRunner

from stitchem.cli import benchmark, stitch

def test_benchmark_cli():
    runner = CliRunner()
    result = runner.invoke(benchmark, "./data/belt")
    assert result.exit_code == 0
    print(result.return_value)
    return

def test_stitch_cli():
    runner = CliRunner()
    result = runner.invoke(stitch, "./data/belt")
    assert result.exit_code == 0
    return



if __name__ == '__main__':
    test_benchmark_cli()