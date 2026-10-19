from astropy.table import Table
url = ("http://exoplanetarchive.ipac.caltech.edu:80/data/ETSS//SuperWASP/TBL/DR1/tile180132/1SWASP_J120939.78+454055.9_lc.tbl")
tab = Table.read(url, format="ipac")
print(len(tab), "measurements")
print(tab.colnames)
print(tab[:3])
tab.write("wasp_lc.csv", overwrite=True)