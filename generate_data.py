"""
generate_data.py
----------------
Generates a SYNTHETIC dataset of aircraft maintenance defect write-ups,
each labelled with its ATA chapter.

The write-ups are built from templates that mimic real technical-log
language: abbreviations (eng, hyd, L/H, IAW AMM), part numbers, positions,
measurements and the usual "found / rectified" phrasing. Vocabulary is
deliberately shared across chapters (leaks, lights, valves, sensors) so the
classification task is not trivially easy.

NOTE: This data is synthetic. State this clearly in your report/slides.

Usage:
    python generate_data.py                # default 3000 rows
    python generate_data.py --n 5000 --seed 7
Output:
    data/defects.csv  (columns: text, ata, chapter_name)
"""
import argparse
import random
import string
from pathlib import Path

import pandas as pd

ATA_NAMES = {
    "21": "Air Conditioning",
    "24": "Electrical Power",
    "25": "Equipment / Furnishings",
    "26": "Fire Protection",
    "27": "Flight Controls",
    "28": "Fuel",
    "29": "Hydraulic Power",
    "30": "Ice & Rain Protection",
    "32": "Landing Gear",
    "33": "Lights",
    "34": "Navigation",
    "36": "Pneumatic",
    "38": "Water / Waste",
    "49": "APU",
    "52": "Doors",
    "72": "Engine",
}

# ---------------------------------------------------------------- vocab
POS = ["L/H", "R/H", "LH", "RH", "left", "right", "#1", "#2", "fwd", "aft",
       "upper", "lower", "inbd", "outbd", "No.1", "No.2"]
FOUND = ["found", "observed", "noticed", "reported", "noted", "obs"]
DURING = ["during walkaround", "during transit check", "during daily check",
          "during preflight", "on arrival", "by crew in flight", "during climb",
          "during taxi", "after landing", "during A-check", "on ground run",
          "during cruise", "during descent", "per PIREP", "by line engineer"]
ACTION = ["replaced", "rectified", "adjusted", "re-secured", "cleaned and re-tested",
          "reset", "repaired", "re-torqued", "changed", "troubleshot and rectified",
          "deferred under MEL", "carried forward"]
REF = ["IAW AMM", "as per AMM", "IAW CMM", "ref AMM", "IAW FIM", "per TSM", "IAW SRM"]
CHECK = ["ops check satisfactory", "ops ck ok", "leak check ok", "test satisfactory",
         "functional test passed", "BITE test ok", "no fault found on retest",
         "system checked serviceable", "found satisfactory", ""]
ACFT = ["VT-ABC", "VT-AXN", "VT-BXK", "VT-RLA", "VT-SYG", "VT-MXQ", "VT-JKL", "VT-TXZ"]

CHAPTERS = {
    "21": {
        "comp": ["pack valve", "ACM", "air cycle machine", "recirc fan", "trim air valve",
                 "pack temp sensor", "duct temp sensor", "mix manifold", "outflow valve",
                 "cabin pressure controller", "ram air door", "pack flow control valve",
                 "zone temp controller", "cabin altitude sensor"],
        "sym": ["cabin too hot", "cabin temp not controlling", "PACK light illuminated",
                "pack trip off", "cabin altitude fluctuating", "duct overheat", "no airflow from gasper",
                "cabin pressure rate high", "AUTO FAIL light on", "bleed trip with pack on",
                "smell in cabin from vents", "outflow valve not modulating"],
    },
    "24": {
        "comp": ["IDG", "generator control unit", "GCU", "TRU", "transformer rectifier unit",
                 "main battery", "battery charger", "static inverter", "bus tie breaker",
                 "standby power module", "ground power receptacle", "AC bus", "DC bus",
                 "external power contactor", "generator feeder cable"],
        "sym": ["GEN OFF BUS light on", "IDG low oil pressure", "battery discharge light on",
                "TRU fail indication", "loss of AC bus", "DRIVE light illuminated",
                "electrical burning smell near E&E bay", "voltage fluctuating",
                "ground power not accepted", "STANDBY PWR OFF light", "IDG disconnect"],
    },
    "25": {
        "comp": ["passenger seat", "seat recline mechanism", "tray table", "overhead bin latch",
                 "lavatory door", "galley oven", "galley cart stopper", "seat belt",
                 "life vest", "cockpit seat", "carpet", "sidewall panel", "PSU panel",
                 "coffee maker", "cabin divider curtain"],
        "sym": ["seat not reclining", "tray table broken", "overhead bin not latching",
                "seat cushion torn", "life vest missing", "galley oven inop",
                "carpet lifting at aisle", "sidewall panel loose", "seat belt buckle damaged",
                "coffee maker not heating", "cart stopper broken", "PSU panel hanging"],
    },
    "26": {
        "comp": ["engine fire loop", "cargo smoke detector", "lav smoke detector",
                 "fire extinguisher bottle", "squib", "APU fire detector", "fire control panel",
                 "wheel well fire loop", "portable extinguisher", "halon bottle"],
        "sym": ["FIRE warning on test failed", "cargo smoke detector fault",
                "lav smoke detector chirping", "bottle pressure low", "squib test fail",
                "DETECTOR FAULT light", "spurious fire warning", "fire loop open circuit",
                "portable extinguisher gauge in red"],
    },
    "27": {
        "comp": ["aileron", "elevator", "rudder", "spoiler", "flap", "slat", "trim actuator",
                 "flap transmission", "PCU", "power control unit", "yaw damper",
                 "stab trim motor", "flap position transmitter", "speed brake lever",
                 "control column", "aileron cable"],
        "sym": ["flap asymmetry", "FLAP disagree", "speed brake not stowing", "yaw damper disengaged",
                "stab trim runaway", "rudder pedal stiff", "SPEEDBRAKE DO NOT ARM light",
                "aileron control heavy", "LE FLAPS TRANSIT light stays on",
                "trim wheel binding", "FEEL DIFF PRESS light", "spoiler not retracting"],
    },
    "28": {
        "comp": ["fuel boost pump", "fuel quantity indicator", "crossfeed valve",
                 "fuel shutoff valve", "fuel quantity probe", "refuel valve", "fuel filler cap",
                 "surge tank", "fuel temp sensor", "fuel quantity processor", "dry bay",
                 "fuel tank access panel", "defuel valve"],
        "sym": ["fuel leak from wing tank", "LOW PRESSURE light on boost pump",
                "fuel quantity indication blank", "fuel imbalance", "crossfeed valve not opening",
                "refuelling overflow from surge tank", "fuel seepage from access panel",
                "FQIS fault", "fuel smell in cabin", "CONFIG light for fuel", "fuel temp indication erratic"],
    },
    "29": {
        "comp": ["EDP", "engine driven pump", "ACMP", "electric motor driven pump",
                 "hyd reservoir", "PTU", "hyd return filter", "case drain filter",
                 "hyd pressure transmitter", "standby hyd system", "hyd line", "accumulator",
                 "hyd quantity transmitter", "hyd supply shutoff valve"],
        "sym": ["hyd leak", "low hyd pressure", "hyd qty low", "HYD PUMP LOW PRESSURE light",
                "OVERHEAT light hyd pump", "hydraulic fluid dripping", "hyd pressure fluctuating",
                "filter bypass indicator popped", "PTU operating abnormally",
                "hyd fluid traces on fuselage belly", "system A pressure drops", "system B qty decreasing"],
    },
    "30": {
        "comp": ["windshield heat", "wing anti-ice valve", "engine anti-ice valve",
                 "pitot heater", "probe heat", "wiper motor", "window heat controller",
                 "rain repellent system", "TAT probe heater", "AOA heater", "drain mast heater"],
        "sym": ["WINDOW HEAT inop", "wiper not parking", "PITOT light illuminated",
                "wing anti-ice valve disagree", "COWL VALVE OPEN light stays on",
                "windshield delamination", "probe heat fail", "wiper streaking",
                "ice detected but WAI not on", "OVERHEAT light window heat"],
    },
    "32": {
        "comp": ["main gear tire", "nose wheel tire", "brake unit", "brake wear pin",
                 "nose wheel steering", "gear downlock", "shock strut", "oleo", "torque link",
                 "antiskid transducer", "wheel speed transducer", "gear door", "uplock actuator",
                 "brake accumulator", "autobrake module", "proximity sensor"],
        "sym": ["tire worn to limits", "tire cut found", "brake wear pin flush",
                "shock strut low extension", "gear not retracting", "gear unsafe light",
                "ANTISKID INOP light", "nose wheel shimmy", "AUTO BRAKE DISARM light",
                "hyd leak from brake unit", "steering not centering", "flat spot on tire",
                "oleo servicing required", "gear door not closing"],
    },
    "33": {
        "comp": ["landing light", "taxi light", "beacon light", "strobe light", "logo light",
                 "nav light", "reading light", "cabin light", "emergency light",
                 "exit sign", "wing illumination light", "runway turnoff light",
                 "panel lighting", "dome light", "emergency exit path lighting"],
        "sym": ["landing light inop", "beacon not flashing", "reading light u/s",
                "cabin light flickering", "emergency light battery low", "exit sign not illuminating",
                "nav light lens cracked", "strobe inop", "panel lights dim", "logo light bulb fused",
                "floor path lighting dim"],
    },
    "34": {
        "comp": ["ADIRU", "IRS", "air data computer", "pitot probe", "static port", "weather radar",
                 "radio altimeter", "TCAS", "EGPWS", "ILS receiver", "VOR receiver", "DME",
                 "GPS antenna", "FMC", "transponder", "standby altimeter", "AOA sensor"],
        "sym": ["IAS disagree", "ALT disagree", "IRS fault light", "radio altimeter fluctuating",
                "TCAS fail", "GPWS INOP", "weather radar no display", "FMC message VERIFY POSITION",
                "ILS localizer flag", "DME no signal", "transponder fail", "AOA disagree",
                "standby altimeter sticking", "GPS position lost"],
    },
    "36": {
        "comp": ["bleed air valve", "PRSOV", "high stage valve", "high stage regulator",
                 "precooler", "precooler control valve", "bleed air regulator", "isolation valve",
                 "bleed duct", "bleed temp sensor", "overpressure switch", "duct clamp"],
        "sym": ["BLEED TRIP OFF light", "DUAL BLEED light", "duct pressure low",
                "bleed overtemp", "bleed duct leak", "WING-BODY OVERHEAT light", "bleed pressure fluctuating",
                "isolation valve disagree", "high stage valve stuck open", "bleed duct clamp loose"],
    },
    "38": {
        "comp": ["lavatory toilet", "vacuum toilet", "waste tank", "potable water tank",
                 "water heater", "lav faucet", "galley water tap", "drain valve",
                 "water quantity indicator", "flush valve", "vacuum blower", "water service panel"],
        "sym": ["toilet not flushing", "waste tank full indication", "no water at galley",
                "water leak under lav", "lav faucet dripping", "water heater inop",
                "potable water qty indicator wrong", "drain valve leaking", "blocked toilet",
                "odour from waste system", "water service panel leak"],
    },
    "49": {
        "comp": ["APU", "APU starter", "APU generator", "APU ECU", "APU fuel control unit",
                 "APU inlet door", "APU oil cooler", "APU EGT thermocouple", "APU igniter",
                 "APU bleed valve", "APU exhaust duct", "APU oil filter"],
        "sym": ["APU not starting", "APU auto shutdown", "APU EGT high", "APU FAULT light",
                "APU LOW OIL PRESSURE light", "APU inlet door not closing", "APU bleed low",
                "APU oil leak", "APU overspeed shutdown", "APU hung start", "APU MAINT light on"],
    },
    "52": {
        "comp": ["passenger door", "service door", "cargo door", "fwd entry door",
                 "aft entry door", "overwing exit", "door seal", "escape slide", "door handle",
                 "cockpit door", "door proximity sensor", "cargo door actuator", "door girt bar",
                 "slide pressure gauge"],
        "sym": ["door seal damaged", "DOOR light illuminated", "cargo door not closing",
                "slide bottle pressure low", "door handle stiff", "cockpit door not locking",
                "door warning in flight", "overwing exit seal torn", "door rigging out",
                "door hinge noise", "girt bar not engaging"],
    },
    "72": {
        "comp": ["fan blade", "HPT blade", "HPC blade", "LPT blade", "combustor",
                 "fan cowl", "thrust reverser", "fuel nozzle", "igniter", "EGT thermocouple",
                 "oil scavenge pump", "engine mount", "N1 sensor", "N2 sensor",
                 "magnetic chip detector", "oil filter", "inlet cowl", "core cowl", "fan case",
                 "VBV actuator", "EEC"],
        "sym": ["EGT exceeded limits", "EGT high on takeoff", "N1 vibration high",
                "bird strike on fan", "fan blade nick found", "oil consumption high",
                "chip light illuminated", "engine surge", "flameout", "REVERSER light on",
                "oil filter bypass", "FOD damage to fan", "borescope findings on HPT",
                "engine slow to start", "engine oil leak", "ENG CONTROL light",
                "N2 vibration high", "fan cowl latch open", "oil pressure fluctuating"],
    },
}

# Vague, generic write-ups: in real tech logs many entries only say "valve leaking"
# or "connector loose", and the only chapter clue is WHERE on the aircraft it is.
# Zones overlap between chapters on purpose, so these entries are genuinely hard.
GENERIC_COMP = ["valve", "sensor", "connector", "wiring harness", "clamp", "seal", "gasket",
                "switch", "indicator", "relay", "line", "access panel", "bracket", "duct", "fitting"]
GENERIC_SYM = ["leaking", "chafed", "loose", "intermittent", "corroded", "inop", "cracked",
               "giving spurious indication", "found damaged", "not operating", "missing screws"]
ZONES = {
    "21": ["mix bay", "pack bay", "cabin ceiling", "aft pressure bulkhead"],
    "24": ["E&E bay", "electronics bay", "flight deck overhead panel", "nacelle"],
    "25": ["cabin", "galley area", "flight deck", "aft cabin"],
    "26": ["aft cargo", "fwd cargo", "nacelle", "APU compartment"],
    "27": ["wing trailing edge", "tail section", "wing leading edge", "flight deck"],
    "28": ["wing root", "center tank area", "wing lower surface", "nacelle"],
    "29": ["wheel well", "wing root", "tail section", "keel beam area"],
    "30": ["flight deck windows", "wing leading edge", "nose section", "nacelle"],
    "32": ["wheel well", "main gear bay", "nose gear bay", "keel beam area"],
    "33": ["cabin ceiling", "wing tip", "tail section", "flight deck overhead panel"],
    "34": ["E&E bay", "nose section", "radome", "flight deck"],
    "36": ["pylon", "wing root", "pack bay", "nacelle"],
    "38": ["aft lav", "galley area", "aft cargo", "fwd lav"],
    "49": ["APU compartment", "tail section", "aft fuselage", "APU bay"],
    "52": ["fwd entry area", "aft cargo", "fwd cargo", "aft cabin"],
    "72": ["nacelle", "pylon", "engine core area", "fan case area"],
}

UNITS = {
    "psi": (100, 3200), "degC": (40, 980), "mm": (1, 25), "%": (1, 105),
    "qts": (1, 20), "kg": (50, 9000), "units": (1, 10),
}


def part_number(rng):
    patterns = [
        lambda: f"{rng.randint(100000, 999999)}-{rng.randint(1, 9)}",
        lambda: f"{rng.randint(1000, 9999)}M{rng.randint(10, 99)}P{rng.randint(1, 20):02d}",
        lambda: f"{rng.choice(string.ascii_uppercase)}{rng.randint(10000, 99999)}-{rng.randint(1, 30):02d}",
        lambda: f"{rng.randint(10, 99)}{rng.choice(string.ascii_uppercase)}{rng.randint(1000, 9999)}",
    ]
    return rng.choice(patterns)()


def serial_number(rng):
    return f"{rng.choice(['SN', 'S/N'])} {rng.choice(string.ascii_uppercase)}{rng.randint(10000, 999999)}"


def measurement(rng):
    unit, (lo, hi) = rng.choice(list(UNITS.items()))
    return f"{rng.randint(lo, hi)} {unit}"


def add_typos(text, rng, rate=0.08):
    """Drop / swap / double a letter in some words, like hurried handwritten entries."""
    words = text.split()
    for i, w in enumerate(words):
        if len(w) > 3 and w.isalpha() and rng.random() < rate:
            j = rng.randrange(1, len(w) - 1)
            op = rng.choice(["drop", "swap", "double"])
            if op == "drop":
                w = w[:j] + w[j + 1:]
            elif op == "swap":
                w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            else:
                w = w[:j] + w[j] + w[j:]
            words[i] = w
    return " ".join(words)


def make_generic(ata, rng):
    comp, sym, zone = rng.choice(GENERIC_COMP), rng.choice(GENERIC_SYM), rng.choice(ZONES[ata])
    templates = [
        f"{comp} {sym} in {zone}",
        f"{rng.choice(POS)} {comp} {sym} at {zone} {rng.choice(DURING)}",
        f"{comp} in {zone} {sym}, {rng.choice(ACTION)} {rng.choice(REF)}",
        f"{rng.choice(FOUND)} {comp} {sym} near {zone}",
    ]
    return rng.choice(templates)


def make_writeup(ata, rng):
    if rng.random() < 0.18:
        text = make_generic(ata, rng)
        if rng.random() < 0.35:
            text = add_typos(text, rng)
        return " ".join(text.split()).strip(" .,")

    c = CHAPTERS[ata]
    comp = rng.choice(c["comp"])
    sym = rng.choice(c["sym"])
    pos = rng.choice(POS)
    # Only some write-ups quote the AMM task number (which contains the chapter);
    # keeping it rare stops the model from just reading the label off the reference.
    if rng.random() < 0.2:
        ref = f"{rng.choice(REF)} {ata}-{rng.randint(0, 99):02d}-{rng.randint(0, 99):02d}"
    else:
        ref = rng.choice(REF)
    pn = part_number(rng)

    templates = [
        f"{sym} {rng.choice(DURING)}. {pos} {comp} {rng.choice(ACTION)} {ref}. {rng.choice(CHECK)}",
        f"{pos} {comp} {sym} {rng.choice(FOUND)} {rng.choice(DURING)}",
        f"{rng.choice(FOUND).capitalize()} {sym}. {comp} P/N {pn} {rng.choice(ACTION)}. {rng.choice(CHECK)}",
        f"{sym} on {rng.choice(ACFT)}. Troubleshooting {ref} carried out, {comp} {rng.choice(ACTION)}",
        f"{pos} {comp} {rng.choice(['u/s', 'inop', 'faulty', 'damaged', 'worn'])}. "
        f"Removed P/N {pn} {serial_number(rng)}, installed new. {rng.choice(CHECK)}",
        f"Crew reported {sym}. {comp} reading {measurement(rng)}. {rng.choice(ACTION)} {ref}",
        f"{sym}",
        f"{comp} {rng.choice(['replaced', 'changed'])} due to {sym}",
        f"{rng.choice(ACFT)} {pos} {comp}: {sym} {rng.choice(DURING)}, {rng.choice(ACTION)}",
    ]
    text = rng.choice(templates)

    # light noise, like real tech logs
    if rng.random() < 0.35:
        text = add_typos(text, rng)
    if rng.random() < 0.15:
        text = text.upper()
    if rng.random() < 0.10:
        text = text.replace(".", "").replace(",", "")
    return " ".join(text.split()).strip(" .,") + ("." if rng.random() < 0.5 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/defects.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    chapters = list(CHAPTERS)
    # slightly imbalanced, like real fleets (engine & landing gear write-ups dominate)
    weights = [3 if a in ("72", "32", "29", "34") else 2 if a in ("21", "24", "27", "28", "33", "36", "49", "52") else 1
               for a in chapters]

    rows, seen = [], set()
    while len(rows) < args.n:
        ata = rng.choices(chapters, weights=weights)[0]
        t = make_writeup(ata, rng)
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        rows.append({"text": t, "ata": ata, "chapter_name": ATA_NAMES[ata]})

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    print(df["ata"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
