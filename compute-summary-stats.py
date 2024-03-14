import lamindb as ln

ln.settings.transform.stem_uid = "beoXrr2SDS7R"
ln.settings.transform.version = "1"

if __name__ == "__main__":

    ln.track()

    df = ln.Artifact.filter(uid="WDNVolxzqPiZ2Mtus9vJ").one().load()
    print(df)

    summary_stats = df.groupby("method").mean()
    print(summary_stats)
