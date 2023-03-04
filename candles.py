import json
import decimal

# candles = []

# for _ in range(10):
#     candles.append({'open': 12, 'high': 12, 'low': 12, 'close': 12, 'time': 12})

# with open(f'candles_temp.dat', 'w') as outfile:
#         json.dump(candles, outfile)

vals = []

def float_to_string(number, precision=20):
    return '{0:.{prec}f}'.format(
        decimal.Context(prec=100).create_decimal(str(number)),
        prec=precision,
    ).rstrip('0').rstrip('.') or '0'


# minimal
def minimal():
    with open(f'candles_ETHUSDT_1H.dat') as data:
        candles = json.load(data)
        for candle in candles:
            vals.append(candle["close"])
        dat = []
        prev_val = vals[0]
        for c in vals:
            dat.append(10 * (c/prev_val - 1))
            prev_val = c
        sdat = []
        for c in dat:
            sdat.append(float_to_string(c))

        with open(f'candles_float.h', 'w') as outfile:
            json.dump(sdat, outfile)


# full
def full():
    with open(f'candles_ETHUSDT_1H.dat') as data:
        candles = json.load(data)
        for i in range(1, len(candles)):
            temp = []
            temp.append(float_to_string(10 * ((candles[i]["open"]  / candles[i-1]["close"]) - 1)))
            temp.append(float_to_string(10 * ((candles[i]["high"]  / candles[i]["open"]) - 1)))
            temp.append(float_to_string(10 * ((candles[i]["low"]   / candles[i]["open"]) - 1)))
            temp.append(float_to_string(10 * ((candles[i]["close"] / candles[i]["open"]) - 1)))
            temp.append(float_to_string(10 * ((candles[i]["close"] / candles[i-1]["close"]) - 1)))
            vals.append(temp)

        with open(f'candles_float.h', 'w') as outfile:
            json.dump(vals, outfile)

full()
# minimal()
