from pnsea import NSE

nse = NSE()

data = nse.options.option_chain("NIFTY")

option_chain = data[0]

print("Columns:")
print(option_chain.columns)

print("\nFirst 5 rows:")
print(option_chain.head())