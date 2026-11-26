"""Metabolic pathway library, structural analysis and simulation.

Stoichiometry follows standard textbook pathways. Rate constants are illustrative, not fitted to
measurements, so structural results (conservation laws, deficiency, sign-consistency, which theorems
apply) are meaningful while specific concentrations and timings are not.
"""
import numpy as np

PATHWAYS = {}


def _p(pid, name, description, species, clamped, reactions, reference):
    PATHWAYS[pid] = {"id": pid, "name": name, "description": description, "species": species,
                     "clamped": clamped, "reactions": reactions, "reference": reference}


def R(rid, name, enzyme, subs, prods, kcat=1.0, km=0.5, reversible=False, regulators=(), deficiency=None, drugs=()):
    return {"id": rid, "name": name, "enzyme": enzyme, "substrates": subs, "products": prods,
            "kcat": kcat, "km": km, "reversible": reversible, "regulators": list(regulators),
            "deficiency": deficiency, "drugs": list(drugs)}


_p("glycolysis", "Glycolysis (Embden-Meyerhof-Parnas)",
   "Glucose to pyruvate in ten enzyme steps, with ATP use and NADH reoxidation attached so the "
   "pathway can reach a steady state. Reads as: glucose to pyruvate, spending two ATP and recovering four. Includes the classic "
   "allosteric controls: ATP inhibits phosphofructokinase and pyruvate kinase, AMP relieves it. AMP is held fixed as a regulator rather than tracked, so the adenine pool here is ATP plus ADP.",
   ["Glc", "G6P", "F6P", "F16BP", "DHAP", "GAP", "BPG", "3PG", "2PG", "PEP", "Pyr",
    "ATP", "ADP", "AMP", "NAD", "NADH", "Pi"],
   ["Glc", "Pi", "AMP"],
   [R("hk", "Hexokinase", "HK", {"Glc": 1, "ATP": 1}, {"G6P": 1, "ADP": 1}, kcat=0.5, km=0.3,
      regulators=[{"species": "G6P", "effect": "inhibit", "k": 1.5}]),
    R("pgi", "Phosphoglucose isomerase", "PGI", {"G6P": 1}, {"F6P": 1}, kcat=2.0, km=0.6, reversible=True),
    R("pfk", "Phosphofructokinase", "PFK", {"F6P": 1, "ATP": 1}, {"F16BP": 1, "ADP": 1}, kcat=2.5, km=0.3,
      regulators=[{"species": "ATP", "effect": "inhibit", "k": 2.0}, {"species": "AMP", "effect": "activate", "k": 0.3}]),
    R("ald", "Aldolase", "ALDO", {"F16BP": 1}, {"DHAP": 1, "GAP": 1}, kcat=1.5, km=0.6, reversible=True),
    R("tpi", "Triose phosphate isomerase", "TPI", {"DHAP": 1}, {"GAP": 1}, kcat=4.0, km=0.6, reversible=True),
    R("gapdh", "Glyceraldehyde-3-P dehydrogenase", "GAPDH", {"GAP": 1, "NAD": 1, "Pi": 1}, {"BPG": 1, "NADH": 1},
      kcat=3.0, km=0.6, reversible=True),
    R("pgk", "Phosphoglycerate kinase", "PGK", {"BPG": 1, "ADP": 1}, {"3PG": 1, "ATP": 1}, kcat=3.0, km=0.6, reversible=True),
    R("pgm", "Phosphoglycerate mutase", "PGM", {"3PG": 1}, {"2PG": 1}, kcat=2.0, km=0.6, reversible=True),
    R("eno", "Enolase", "ENO", {"2PG": 1}, {"PEP": 1}, kcat=2.0, km=0.6, reversible=True),
    R("pk", "Pyruvate kinase", "PK", {"PEP": 1, "ADP": 1}, {"Pyr": 1, "ATP": 1}, kcat=3.0, km=0.3,
      regulators=[{"species": "ATP", "effect": "inhibit", "k": 5.0}, {"species": "F16BP", "effect": "activate", "k": 0.4}]),
    R("atpase", "ATP consumption (cell work)", "ATPase", {"ATP": 1}, {"ADP": 1, "Pi": 1}, kcat=1.2, km=0.5),
    R("pdh", "Pyruvate consumption (PDH or export)", "PDH", {"Pyr": 1}, {}, kcat=1.5, km=0.4),
    R("nox", "NADH reoxidation (respiration or fermentation)", "NOX", {"NADH": 1}, {"NAD": 1}, kcat=1.2, km=0.3),
    R("oxphos", "ATP regeneration from NADH (oxidative phosphorylation)", "OXPHOS", {"NADH": 1, "ADP": 1, "Pi": 1},
      {"NAD": 1, "ATP": 1}, kcat=1.5, km=0.3)],
   "Berg, Tymoczko and Stryer, Biochemistry, glycolysis chapter")

_p("fermentation", "Lactate fermentation",
   "The anaerobic branch: lactate dehydrogenase regenerates NAD from NADH so glycolysis can keep running "
   "without oxygen. This is the Warburg-style branch point in tumour metabolism.",
   ["Pyr", "Lac", "NAD", "NADH", "H"], ["H", "Pyr"],
   [R("ldh", "Lactate dehydrogenase", "LDH", {"Pyr": 1, "NADH": 1}, {"Lac": 1, "NAD": 1}, kcat=2.0, km=0.3, reversible=True),
    R("mct", "Lactate export", "MCT", {"Lac": 1}, {}, kcat=1.0, km=0.6),
    R("supply", "NADH supply from glycolysis (GAPDH upstream)", "GAPDH", {"NAD": 1}, {"NADH": 1}, kcat=0.6, km=0.4)],
   "Standard anaerobic glycolysis branch")

_p("ppp", "Pentose phosphate pathway (oxidative branch)",
   "The branch that makes NADPH for antioxidant defence and biosynthesis. Cells shifting flux here is one "
   "known route to surviving oxidative stress and some drugs.",
   ["G6P", "6PGL", "6PG", "Ru5P", "CO2", "NADP", "NADPH"], ["CO2", "G6P"],
   [R("g6pd", "Glucose-6-P dehydrogenase", "G6PD", {"G6P": 1, "NADP": 1}, {"6PGL": 1, "NADPH": 1}, kcat=1.5, km=0.3,
      regulators=[{"species": "NADPH", "effect": "inhibit", "k": 1.0}]),
    R("pgls", "6-phosphogluconolactonase", "PGLS", {"6PGL": 1}, {"6PG": 1}, kcat=3.0, km=0.4),
    R("pgd", "6-phosphogluconate dehydrogenase", "PGD", {"6PG": 1, "NADP": 1}, {"Ru5P": 1, "CO2": 1, "NADPH": 1},
      kcat=1.5, km=0.3),
    R("nadph_use", "NADPH consumption (biosynthesis and antioxidant defence)", "NADPH-use", {"NADPH": 1}, {"NADP": 1},
      kcat=1.0, km=0.3),
    R("r5p_use", "Ribose-5-P consumption (nucleotide synthesis)", "R5P-use", {"Ru5P": 1}, {}, kcat=1.0, km=0.3)],
   "Berg, Tymoczko and Stryer, pentose phosphate pathway chapter")

_p("tca", "Citric acid cycle (TCA)",
   "The cycle that oxidises acetyl-CoA to CO2, feeding NADH to the respiratory chain. Its conserved cycle "
   "structure makes it a good test of conservation-law analysis.",
   ["Pyr", "AcCoA", "OAA", "Cit", "IsoCit", "AKG", "SucCoA", "Suc", "Fum", "Mal", "CoA", "NAD", "NADH", "CO2",
    "GDP", "GTP", "Pi", "PEP", "ADP", "ATP"],
   ["CO2", "Pi", "GDP", "Pyr", "ADP"],
   [R("pdh", "Pyruvate dehydrogenase (entry into the cycle)", "PDH", {"Pyr": 1, "CoA": 1, "NAD": 1},
      {"AcCoA": 1, "NADH": 1, "CO2": 1}, kcat=1.2, km=0.3,
      regulators=[{"species": "NADH", "effect": "inhibit", "k": 1.0}]),
    R("cs", "Citrate synthase", "CS", {"AcCoA": 1, "OAA": 1}, {"Cit": 1, "CoA": 1}, kcat=1.5, km=0.3,
      regulators=[{"species": "NADH", "effect": "inhibit", "k": 1.0}]),
    R("acon", "Aconitase", "ACO", {"Cit": 1}, {"IsoCit": 1}, kcat=2.0, km=0.4, reversible=True),
    R("idh", "Isocitrate dehydrogenase", "IDH", {"IsoCit": 1, "NAD": 1}, {"AKG": 1, "NADH": 1, "CO2": 1}, kcat=1.2, km=0.3,
      regulators=[{"species": "NADH", "effect": "inhibit", "k": 0.8}]),
    R("kgdh", "Alpha-ketoglutarate dehydrogenase", "KGDH", {"AKG": 1, "NAD": 1, "CoA": 1}, {"SucCoA": 1, "NADH": 1, "CO2": 1},
      kcat=1.2, km=0.3),
    R("scs", "Succinyl-CoA synthetase", "SCS", {"SucCoA": 1, "GDP": 1, "Pi": 1}, {"Suc": 1, "CoA": 1, "GTP": 1}, kcat=1.5, km=0.3),
    R("sdh", "Succinate dehydrogenase", "SDH", {"Suc": 1}, {"Fum": 1}, kcat=1.5, km=0.4, reversible=True),
    R("fum", "Fumarase", "FUM", {"Fum": 1}, {"Mal": 1}, kcat=2.5, km=0.4, reversible=True),
    R("mdh", "Malate dehydrogenase", "MDH", {"Mal": 1, "NAD": 1}, {"OAA": 1, "NADH": 1}, kcat=1.5, km=0.4, reversible=True),
    R("etc", "Respiration: NADH reoxidation with ATP synthesis", "ETC", {"NADH": 1, "ADP": 1, "Pi": 1},
      {"NAD": 1, "ATP": 1}, kcat=2.0, km=0.3),
    R("gtp_use", "GTP consumption", "GTP-use", {"GTP": 1}, {"GDP": 1}, kcat=1.5, km=0.3),
    R("atp_use", "ATP consumption (cell work)", "ATPase", {"ATP": 1}, {"ADP": 1, "Pi": 1}, kcat=1.2, km=0.5),
    R("pepck", "Phosphoenolpyruvate carboxykinase (exit to gluconeogenesis)", "PEPCK",
      {"OAA": 1}, {"PEP": 1, "CO2": 1}, kcat=1.2, km=0.4),
    R("pep_out", "PEP into gluconeogenesis", "Transfer", {"PEP": 1}, {}, kcat=1.5, km=0.4)],
   "Berg, Tymoczko and Stryer, citric acid cycle chapter")


_p("urea", "Urea cycle",
   "How the body turns toxic ammonia into urea for excretion. Every step has a known inherited deficiency, and "
   "blocking any of them backs ammonia up into the blood.",
   ["NH3", "CO2", "CP", "Orn", "Citrul", "Asp", "ASA", "Arg", "Urea", "Fum"], ["CO2", "Asp"],
   [R("nh3_in", "Ammonia from protein turnover", "Supply", {}, {"NH3": 1}, kcat=0.3),
    R("cps1", "Carbamoyl phosphate synthetase I", "CPS1", {"NH3": 1, "CO2": 1}, {"CP": 1}, kcat=0.6, km=0.4,
      deficiency="CPS1 deficiency: ammonia builds up from birth, the most severe urea cycle disorder"),
    R("otc", "Ornithine transcarbamylase", "OTC", {"CP": 1, "Orn": 1}, {"Citrul": 1}, kcat=1.5, km=0.3,
      deficiency="OTC deficiency: the commonest urea cycle disorder, X-linked; carbamoyl phosphate spills into orotic acid"),
    R("ass", "Argininosuccinate synthetase", "ASS", {"Citrul": 1, "Asp": 1}, {"ASA": 1}, kcat=1.2, km=0.3,
      deficiency="Citrullinemia type I: citrulline accumulates"),
    R("asl", "Argininosuccinate lyase", "ASL", {"ASA": 1}, {"Arg": 1, "Fum": 1}, kcat=1.5, km=0.3,
      deficiency="Argininosuccinic aciduria: argininosuccinate accumulates"),
    R("arg1", "Arginase 1", "ARG1", {"Arg": 1}, {"Urea": 1, "Orn": 1}, kcat=1.5, km=0.3,
      deficiency="Argininemia: arginine accumulates, with spastic diplegia rather than acute ammonia crises"),
    R("urea_out", "Urea excretion", "Excretion", {"Urea": 1}, {}, kcat=2.0, km=0.4),
    R("fum_out", "Fumarate to the TCA cycle", "Transfer", {"Fum": 1}, {}, kcat=2.0, km=0.4)],
   "Standard urea cycle; deficiencies as described in clinical genetics texts")

_p("phe", "Phenylalanine and tyrosine catabolism",
   "The route that breaks down phenylalanine. Blocks along it cause several of the best known inherited metabolic "
   "diseases, and one of them is treated with a drug that deliberately blocks a step further down.",
   ["Phe", "Tyr", "HPP", "HGA", "MAA", "Fum", "AcAc"], [],
   [R("phe_in", "Dietary phenylalanine", "Diet", {}, {"Phe": 1}, kcat=0.5),
    R("pah", "Phenylalanine hydroxylase", "PAH", {"Phe": 1}, {"Tyr": 1}, kcat=1.2, km=0.4,
      deficiency="Phenylketonuria (PKU): phenylalanine accumulates and harms the developing brain; treated by dietary restriction",
      drugs=[{"name": "Sapropterin (BH4)", "effect": "activates", "note": "cofactor analogue that boosts residual enzyme in some patients"}]),
    R("tat", "Tyrosine aminotransferase", "TAT", {"Tyr": 1}, {"HPP": 1}, kcat=1.5, km=0.4,
      deficiency="Tyrosinemia type II: tyrosine accumulates, with eye and skin lesions"),
    R("hpd", "4-hydroxyphenylpyruvate dioxygenase", "HPD", {"HPP": 1}, {"HGA": 1}, kcat=1.5, km=0.4,
      deficiency="Tyrosinemia type III",
      drugs=[{"name": "Nitisinone", "effect": "inhibits", "note": "blocks this step on purpose to stop toxic metabolites forming further down in tyrosinemia type I"}]),
    R("hgd", "Homogentisate 1,2-dioxygenase", "HGD", {"HGA": 1}, {"MAA": 1}, kcat=1.5, km=0.4,
      deficiency="Alkaptonuria: homogentisic acid accumulates, darkens urine and damages cartilage"),
    R("fah", "Fumarylacetoacetate hydrolase", "FAH", {"MAA": 1}, {"Fum": 1, "AcAc": 1}, kcat=1.5, km=0.4,
      deficiency="Tyrosinemia type I: toxic intermediates build up and damage liver and kidney"),
    R("fum_out", "Fumarate to the TCA cycle", "Transfer", {"Fum": 1}, {}, kcat=2.0, km=0.4),
    R("acac_out", "Acetoacetate to ketone metabolism", "Transfer", {"AcAc": 1}, {}, kcat=2.0, km=0.4)],
   "Phenylalanine to fumarate and acetoacetate; diseases as in clinical genetics texts")

_p("galactose", "Galactose metabolism (Leloir pathway)",
   "How milk sugar enters glycolysis. A block at the second step is classic galactosemia, where galactose-1-phosphate "
   "accumulates inside cells.",
   ["Gal", "Gal1P", "UDPGlc", "UDPGal", "G1P", "G6P", "ATP", "ADP"], ["ATP", "ADP"],
   [R("gal_in", "Dietary galactose (lactose)", "Diet", {}, {"Gal": 1}, kcat=0.15),
    R("galk", "Galactokinase", "GALK", {"Gal": 1, "ATP": 1}, {"Gal1P": 1, "ADP": 1}, kcat=0.6, km=0.3,
      deficiency="Galactokinase deficiency: galactose accumulates and forms cataracts, but without the systemic illness of classic galactosemia"),
    R("galt", "Galactose-1-phosphate uridylyltransferase", "GALT", {"Gal1P": 1, "UDPGlc": 1}, {"G1P": 1, "UDPGal": 1},
      kcat=1.5, km=0.3,
      deficiency="Classic galactosemia: galactose-1-phosphate accumulates and is toxic to liver, brain and ovary"),
    R("gale", "UDP-galactose 4-epimerase", "GALE", {"UDPGal": 1}, {"UDPGlc": 1}, kcat=2.0, km=0.4, reversible=True,
      deficiency="Epimerase deficiency galactosemia, usually milder"),
    R("pgm", "Phosphoglucomutase", "PGM", {"G1P": 1}, {"G6P": 1}, kcat=2.0, km=0.4, reversible=True),
    R("g6p_out", "Glucose-6-phosphate into glycolysis", "Transfer", {"G6P": 1}, {}, kcat=1.5, km=0.4)],
   "Leloir pathway; galactosemias as in clinical genetics texts")

_p("purine", "Purine salvage and degradation",
   "Recycling purines instead of making them from scratch, and the breakdown route that ends in uric acid. It holds "
   "two famous enzyme deficiencies and the target of the commonest gout drug.",
   ["PRPP", "Hx", "IMP", "Xan", "Urate", "Ado", "Ino"], ["PRPP", "Ado"],
   [R("hgprt", "Hypoxanthine-guanine phosphoribosyltransferase", "HGPRT", {"Hx": 1, "PRPP": 1}, {"IMP": 1},
      kcat=1.5, km=0.3,
      deficiency="Lesch-Nyhan syndrome when complete: salvage fails, purines are shunted to uric acid, causing gout and severe neurological disease"),
    R("ada", "Adenosine deaminase", "ADA", {"Ado": 1}, {"Ino": 1}, kcat=1.5, km=0.3,
      deficiency="ADA deficiency: a form of severe combined immunodeficiency; toxic metabolites kill developing lymphocytes",
      drugs=[{"name": "Pentostatin", "effect": "inhibits", "note": "used in hairy cell leukaemia, exploiting the same lymphocyte toxicity"}]),
    R("pnp", "Purine nucleoside phosphorylase", "PNP", {"Ino": 1}, {"Hx": 1}, kcat=2.0, km=0.3,
      deficiency="PNP deficiency: T-cell immunodeficiency"),
    R("xo1", "Xanthine oxidase, first step", "XO", {"Hx": 1}, {"Xan": 1}, kcat=1.0, km=0.4,
      drugs=[{"name": "Allopurinol", "effect": "inhibits", "note": "the standard gout drug, lowering uric acid production"},
             {"name": "Febuxostat", "effect": "inhibits", "note": "a selective alternative"}]),
    R("xo2", "Xanthine oxidase, second step", "XO", {"Xan": 1}, {"Urate": 1}, kcat=1.0, km=0.4,
      drugs=[{"name": "Allopurinol", "effect": "inhibits", "note": "same enzyme, same block"}]),
    R("imp_use", "IMP into nucleotide synthesis", "Transfer", {"IMP": 1}, {}, kcat=1.5, km=0.4),
    R("urate_out", "Uric acid excretion", "Excretion", {"Urate": 1}, {}, kcat=1.0, km=0.5,
      drugs=[{"name": "Probenecid", "effect": "activates", "note": "increases urinary excretion of uric acid"}])],
   "Purine salvage and degradation; diseases and drugs as in clinical pharmacology texts")


_p("glycogen", "Glycogen storage and breakdown",
   "Storing glucose as glycogen and releasing it between meals. Glycogen itself is a store, so its level drifts up "
   "or down rather than settling; everything else in the pathway settles. Blocks here are the glycogen storage diseases, and "
   "they split neatly into liver forms that cause low blood sugar and muscle forms that cause exercise intolerance.",
   ["G6P", "G1P", "UDPGlc", "Glycogen", "Glc", "UTP", "UDP", "Pi"], ["UTP", "Pi", "G6P"],
   [R("pgm", "Phosphoglucomutase", "PGM", {"G6P": 1}, {"G1P": 1}, kcat=1.5, km=0.4, reversible=True),
    R("ugp", "UDP-glucose pyrophosphorylase", "UGP", {"G1P": 1, "UTP": 1}, {"UDPGlc": 1}, kcat=1.4, km=0.3),
    R("gys", "Glycogen synthase", "GYS", {"UDPGlc": 1}, {"Glycogen": 1, "UDP": 1}, kcat=0.4, km=0.3,
      deficiency="Glycogen storage disease 0: too little glycogen, so fasting hypoglycaemia without hepatomegaly"),
    R("pyg", "Glycogen phosphorylase", "PYG", {"Glycogen": 1, "Pi": 1}, {"G1P": 1}, kcat=1.2, km=0.3,
      deficiency="McArdle disease in muscle (PYGM) or Hers disease in liver (PYGL): glycogen cannot be released"),
    R("g6pase", "Glucose-6-phosphatase", "G6Pase", {"G6P": 1}, {"Glc": 1, "Pi": 1}, kcat=2.0, km=0.4,
      deficiency="Von Gierke disease (GSD I): glucose-6-phosphate cannot leave as glucose, causing severe fasting hypoglycaemia, lactic acidosis and hepatomegaly"),
    R("glc_out", "Glucose to the blood", "Export", {"Glc": 1}, {}, kcat=1.5, km=0.4),
    R("udp_recycle", "UDP back to UTP", "NDPK", {"UDP": 1}, {"UTP": 1}, kcat=2.0, km=0.4)],
   "Glycogen metabolism; storage diseases as in clinical genetics texts")

_p("gluconeogenesis", "Gluconeogenesis",
   "Making glucose from pyruvate when the diet does not supply it. Its enzymes are the ones that fail in fasting "
   "hypoglycaemia, and one of them is the site where metformin acts indirectly.",
   ["Pyr", "OAA", "PEP", "F16BP", "F6P", "G6P", "Glc", "CO2", "ATP", "ADP", "Pi"], ["Pyr", "CO2", "ATP", "ADP", "Pi"],
   [R("pc", "Pyruvate carboxylase", "PC", {"Pyr": 1, "CO2": 1, "ATP": 1}, {"OAA": 1, "ADP": 1, "Pi": 1}, kcat=0.8, km=0.3,
      deficiency="Pyruvate carboxylase deficiency: lactic acidosis and failure to make glucose; biotin is its cofactor"),
    R("pepck", "PEP carboxykinase", "PEPCK", {"OAA": 1}, {"PEP": 1, "CO2": 1}, kcat=1.2, km=0.3,
      deficiency="PEPCK deficiency: severe fasting hypoglycaemia"),
    R("enol_rev", "Reverse glycolysis to fructose-1,6-bisphosphate", "Enolase-Aldolase", {"PEP": 1}, {"F16BP": 1},
      kcat=1.2, km=0.4),
    R("fbpase", "Fructose-1,6-bisphosphatase", "FBPase", {"F16BP": 1}, {"F6P": 1, "Pi": 1}, kcat=1.2, km=0.3,
      deficiency="FBPase deficiency: hypoglycaemia and lactic acidosis after fasting or illness"),
    R("pgi", "Phosphoglucose isomerase", "PGI", {"F6P": 1}, {"G6P": 1}, kcat=2.0, km=0.4, reversible=True),
    R("g6pase", "Glucose-6-phosphatase", "G6Pase", {"G6P": 1}, {"Glc": 1, "Pi": 1}, kcat=1.2, km=0.4,
      deficiency="Von Gierke disease (GSD I), the same enzyme that ends glycogen breakdown"),
    R("glc_out", "Glucose to the blood", "Export", {"Glc": 1}, {}, kcat=1.5, km=0.4)],
   "Gluconeogenesis; deficiencies as in clinical genetics texts")

_p("fructose", "Fructose metabolism",
   "How dietary fructose enters glycolysis. A block at the second step traps phosphate inside the liver cell and "
   "causes hereditary fructose intolerance, treated entirely by removing fructose from the diet.",
   ["Fru", "F1P", "DHAP", "Glyceraldehyde", "GAP", "ATP", "ADP"], ["ATP", "ADP"],
   [R("fru_in", "Dietary fructose", "Diet", {}, {"Fru": 1}, kcat=0.35),
    R("khk", "Fructokinase", "KHK", {"Fru": 1, "ATP": 1}, {"F1P": 1, "ADP": 1}, kcat=1.2, km=0.3,
      deficiency="Essential fructosuria: harmless, fructose simply appears in the urine"),
    R("aldob", "Aldolase B", "ALDOB", {"F1P": 1}, {"DHAP": 1, "Glyceraldehyde": 1}, kcat=1.2, km=0.3,
      deficiency="Hereditary fructose intolerance: fructose-1-phosphate accumulates, traps phosphate and blocks glucose production, causing hypoglycaemia and liver damage after fructose"),
    R("tk", "Triose kinase", "TK", {"Glyceraldehyde": 1, "ATP": 1}, {"GAP": 1, "ADP": 1}, kcat=1.5, km=0.3),
    R("dhap_out", "DHAP into glycolysis", "Transfer", {"DHAP": 1}, {}, kcat=1.5, km=0.4),
    R("gap_out", "GAP into glycolysis", "Transfer", {"GAP": 1}, {}, kcat=1.5, km=0.4)],
   "Fructose metabolism; hereditary fructose intolerance as in clinical genetics texts")

_p("betaox", "Fatty acid beta-oxidation",
   "Burning fat for energy, including the carnitine shuttle that carries fatty acids into mitochondria. Blocks here "
   "cause hypoglycaemia during fasting or illness, because the body cannot switch from sugar to fat.",
   ["FA", "AcylCoA", "AcylCarn", "AcylCoA_m", "EnoylCoA", "AcCoA", "NAD", "NADH", "CoA", "Carn", "FAD", "FADH2"],
   ["FA", "Carn"],
   [R("acs", "Acyl-CoA synthetase", "ACS", {"FA": 1, "CoA": 1}, {"AcylCoA": 1}, kcat=1.0, km=0.3),
    R("cpt1", "Carnitine palmitoyltransferase I", "CPT1", {"AcylCoA": 1, "Carn": 1}, {"AcylCarn": 1, "CoA": 1},
      kcat=1.2, km=0.3,
      deficiency="CPT1 deficiency: fat cannot enter mitochondria, causing hypoketotic hypoglycaemia when fasting"),
    R("cpt2", "Carnitine palmitoyltransferase II", "CPT2", {"AcylCarn": 1, "CoA": 1}, {"AcylCoA_m": 1, "Carn": 1},
      kcat=1.5, km=0.3,
      deficiency="CPT2 deficiency: muscle breakdown after exercise or fasting"),
    R("acad", "Acyl-CoA dehydrogenase", "MCAD", {"AcylCoA_m": 1, "FAD": 1}, {"EnoylCoA": 1, "FADH2": 1}, kcat=1.2, km=0.3,
      deficiency="MCAD deficiency: the commonest fat oxidation disorder; fasting illness causes hypoketotic hypoglycaemia and can be fatal, but is prevented simply by avoiding long fasts"),
    R("spiral", "Rest of the beta-oxidation spiral", "HADH-Thiolase", {"EnoylCoA": 1, "NAD": 1, "CoA": 1},
      {"AcCoA": 1, "NADH": 1}, kcat=1.5, km=0.3),
    R("accoa_out", "Acetyl-CoA to the TCA cycle or ketones", "Transfer", {"AcCoA": 1}, {"CoA": 1}, kcat=1.5, km=0.4),
    R("etf", "FADH2 reoxidation", "ETF", {"FADH2": 1}, {"FAD": 1}, kcat=2.0, km=0.3),
    R("nadh_ox", "NADH reoxidation", "Respiration", {"NADH": 1}, {"NAD": 1}, kcat=2.0, km=0.3)],
   "Beta-oxidation and the carnitine shuttle; disorders as in clinical genetics texts")

_p("ketone", "Ketone body production and use",
   "Turning acetyl-CoA into ketone bodies during fasting, and burning them in brain and muscle. This is the backup "
   "fuel that fails when fat oxidation is blocked.",
   ["AcCoA", "AcAcCoA", "HMGCoA", "AcAc", "BHB", "CoA", "NAD", "NADH"], [],
   [R("fat_in", "Acetyl-CoA arriving from fat oxidation", "Beta-oxidation", {"CoA": 1}, {"AcCoA": 1}, kcat=1.6, km=0.3),
    R("thiolase", "Thiolase", "ACAT1", {"AcCoA": 2}, {"AcAcCoA": 1, "CoA": 1}, kcat=0.8, km=0.3,
      deficiency="Beta-ketothiolase deficiency: ketoacidotic crises"),
    R("hmgcs", "HMG-CoA synthase", "HMGCS2", {"AcAcCoA": 1, "AcCoA": 1}, {"HMGCoA": 1, "CoA": 1}, kcat=2.5, km=0.3,
      deficiency="HMG-CoA synthase deficiency: no ketones when fasting, so hypoketotic hypoglycaemia"),
    R("hmgcl", "HMG-CoA lyase", "HMGCL", {"HMGCoA": 1}, {"AcAc": 1, "AcCoA": 1}, kcat=1.5, km=0.3,
      deficiency="HMG-CoA lyase deficiency: no ketones and toxic intermediates accumulate"),
    R("bdh", "Beta-hydroxybutyrate dehydrogenase", "BDH1", {"AcAc": 1, "NADH": 1}, {"BHB": 1, "NAD": 1},
      kcat=1.5, km=0.3, reversible=True),
    R("scot", "Ketone use in tissues", "SCOT", {"AcAc": 1, "CoA": 1}, {"AcAcCoA": 1}, kcat=0.6, km=0.3,
      deficiency="SCOT deficiency: ketones are made but cannot be used, causing permanent ketosis and crises"),
    R("acacoa_use", "Acetoacetyl-CoA burned for energy", "Thiolase", {"AcAcCoA": 1, "CoA": 1}, {"AcCoA": 2},
      kcat=1.2, km=0.3),
    R("bhb_out", "Beta-hydroxybutyrate to the blood", "Export", {"BHB": 1}, {}, kcat=1.5, km=0.4),
    R("acac_out", "Acetoacetate to the blood", "Export", {"AcAc": 1}, {}, kcat=1.5, km=0.4),
    R("nadh_ox", "NADH reoxidation", "Respiration", {"NADH": 1}, {"NAD": 1}, kcat=1.5, km=0.3)],
   "Ketogenesis and ketolysis; disorders as in clinical genetics texts")

_p("bcaa", "Branched-chain amino acid breakdown",
   "Breaking down leucine, isoleucine and valine. A block at the second step is maple syrup urine disease, named for "
   "the smell of the accumulating keto acids.",
   ["BCAA", "BCKA", "BCAcylCoA", "AcCoA", "NAD", "NADH", "CoA"], ["BCAA"],
   [R("bcat", "Branched-chain aminotransferase", "BCAT", {"BCAA": 1}, {"BCKA": 1}, kcat=0.72, km=0.3),
    R("bckdh", "Branched-chain ketoacid dehydrogenase", "BCKDH", {"BCKA": 1, "NAD": 1, "CoA": 1},
      {"BCAcylCoA": 1, "NADH": 1}, kcat=1.6, km=0.3,
      deficiency="Maple syrup urine disease: branched-chain keto acids accumulate and are neurotoxic; thiamine helps in some forms",
      drugs=[{"name": "Thiamine", "effect": "activates", "note": "cofactor; a minority of patients respond to high doses"}]),
    R("ivd", "Isovaleryl-CoA dehydrogenase and the rest", "IVD", {"BCAcylCoA": 1}, {"AcCoA": 1}, kcat=1.5, km=0.3,
      deficiency="Isovaleric acidemia: a sweaty-feet odour and metabolic crises"),
    R("accoa_out", "Acetyl-CoA onward", "Transfer", {"AcCoA": 1}, {"CoA": 1}, kcat=1.5, km=0.4),
    R("nadh_ox", "NADH reoxidation", "Respiration", {"NADH": 1}, {"NAD": 1}, kcat=2.0, km=0.3)],
   "Branched-chain amino acid catabolism; MSUD as in clinical genetics texts")

_p("homocysteine", "Methionine and homocysteine",
   "The cycle that recycles methionine and disposes of homocysteine. Blocks raise homocysteine, which damages blood "
   "vessels, and the treatments are vitamins plus a bypass.",
   ["Met", "SAM", "SAH", "Hcy", "Cystathionine", "Cys", "Ser", "MeTHF", "THF", "B12"], ["Met", "Ser", "B12"],
   [R("mat", "Methionine adenosyltransferase", "MAT", {"Met": 1}, {"SAM": 1}, kcat=1.0, km=0.3),
    R("methyl", "Methyl transfer reactions", "MTs", {"SAM": 1}, {"SAH": 1}, kcat=1.5, km=0.3),
    R("sahh", "SAH hydrolase", "AHCY", {"SAH": 1}, {"Hcy": 1}, kcat=1.5, km=0.3, reversible=True),
    R("ms", "Methionine synthase", "MTR", {"Hcy": 1, "MeTHF": 1, "B12": 1}, {"Met": 1, "THF": 1, "B12": 1},
      kcat=1.2, km=0.3,
      deficiency="Methionine synthase deficiency, or functional loss from B12 deficiency: homocysteine rises and methionine falls",
      drugs=[{"name": "Hydroxocobalamin (B12)", "effect": "activates", "note": "restores the cofactor when the problem is B12 supply"}]),
    R("cbs", "Cystathionine beta-synthase", "CBS", {"Hcy": 1, "Ser": 1}, {"Cystathionine": 1}, kcat=1.2, km=0.3,
      deficiency="Classic homocystinuria: homocysteine accumulates, causing lens dislocation, marfanoid build, clots and stroke",
      drugs=[{"name": "Pyridoxine (B6)", "effect": "activates", "note": "cofactor; about half of patients respond"},
             {"name": "Betaine", "effect": "activates", "note": "opens a bypass that remethylates homocysteine back to methionine"}]),
    R("cth", "Cystathionase", "CTH", {"Cystathionine": 1}, {"Cys": 1}, kcat=1.5, km=0.3),
    R("cys_out", "Cysteine onward", "Transfer", {"Cys": 1}, {}, kcat=1.5, km=0.4),
    R("mthfr", "MTHFR: folate methyl supply", "MTHFR", {"THF": 1}, {"MeTHF": 1}, kcat=1.5, km=0.3,
      deficiency="MTHFR deficiency: less methyl folate, so homocysteine rises",
      drugs=[{"name": "Folate", "effect": "activates", "note": "supports the methyl supply"}])],
   "Methionine cycle and transsulfuration; homocystinurias as in clinical genetics texts")

_p("folate", "Folate and one-carbon metabolism",
   "The carrier of one-carbon units for DNA synthesis. It is the target of methotrexate and of trimethoprim in "
   "bacteria, which makes it a pathway where a deliberate block is the treatment.",
   ["Folate", "DHF", "THF", "MeTHF", "dUMP", "dTMP", "NADPH", "NADP"], ["Folate", "dUMP", "NADPH", "NADP"],
   [R("dhfr1", "Dihydrofolate reductase, first pass", "DHFR", {"Folate": 1, "NADPH": 1}, {"DHF": 1, "NADP": 1},
      kcat=0.3, km=0.3),
    R("dhfr2", "Dihydrofolate reductase", "DHFR", {"DHF": 1, "NADPH": 1}, {"THF": 1, "NADP": 1}, kcat=4.0, km=0.3,
      deficiency="DHFR deficiency: megaloblastic anaemia",
      drugs=[{"name": "Methotrexate", "effect": "inhibits", "note": "blocks this step in cancer, rheumatoid arthritis and psoriasis"},
             {"name": "Trimethoprim", "effect": "inhibits", "note": "selective for the bacterial enzyme"}]),
    R("shmt", "Serine hydroxymethyltransferase", "SHMT", {"THF": 1}, {"MeTHF": 1}, kcat=1.5, km=0.3),
    R("ts", "Thymidylate synthase", "TYMS", {"MeTHF": 1, "dUMP": 1}, {"dTMP": 1, "DHF": 1}, kcat=0.8, km=0.3,
      drugs=[{"name": "5-fluorouracil", "effect": "inhibits", "note": "the classic chemotherapy block of DNA precursor supply"}]),
    R("dtmp_out", "dTMP into DNA synthesis", "Transfer", {"dTMP": 1}, {}, kcat=1.5, km=0.4)],
   "Folate one-carbon metabolism; drug targets as in clinical pharmacology texts")

_p("heme", "Heme synthesis",
   "Building heme from glycine and succinyl-CoA. Partial blocks cause the porphyrias, where intermediates accumulate, "
   "and lead poisoning blocks two steps at once.",
   ["Gly", "SucCoA", "ALA", "PBG", "HMB", "Copro", "Proto", "Heme", "Fe"], ["Gly", "SucCoA", "Fe"],
   [R("alas", "ALA synthase", "ALAS", {"Gly": 1, "SucCoA": 1}, {"ALA": 1}, kcat=0.6, km=0.3,
      regulators=[{"species": "Heme", "effect": "inhibit", "k": 0.5}],
      deficiency="X-linked sideroblastic anaemia when deficient; when other steps are blocked this enzyme is de-repressed and drives the attack",
      drugs=[{"name": "Haem arginate", "effect": "inhibits", "note": "given in acute porphyria to switch this step off by feedback"},
             {"name": "Givosiran", "effect": "inhibits", "note": "silences this enzyme's message in acute hepatic porphyria"}]),
    R("alad", "ALA dehydratase", "ALAD", {"ALA": 2}, {"PBG": 1}, kcat=1.2, km=0.3,
      deficiency="ALAD porphyria, very rare",
      drugs=[{"name": "Lead", "effect": "inhibits", "note": "a poison rather than a drug: lead blocks this step and ferrochelatase, so ALA accumulates"}]),
    R("pbgd", "PBG deaminase", "PBGD", {"PBG": 1}, {"HMB": 1}, kcat=1.2, km=0.3,
      deficiency="Acute intermittent porphyria: attacks of pain, neuropathy and confusion, triggered by drugs, fasting and hormones"),
    R("upd", "Uroporphyrinogen steps", "UROD", {"HMB": 1}, {"Copro": 1}, kcat=1.5, km=0.3,
      deficiency="Porphyria cutanea tarda: blistering photosensitive skin, the commonest porphyria"),
    R("cpox", "Coproporphyrinogen oxidase", "CPOX", {"Copro": 1}, {"Proto": 1}, kcat=1.5, km=0.3,
      deficiency="Hereditary coproporphyria"),
    R("fech", "Ferrochelatase", "FECH", {"Proto": 1, "Fe": 1}, {"Heme": 1}, kcat=1.5, km=0.3,
      deficiency="Erythropoietic protoporphyria: painful sun sensitivity",
      drugs=[{"name": "Lead", "effect": "inhibits", "note": "the second step lead blocks"}]),
    R("heme_use", "Heme into haemoglobin and enzymes", "Transfer", {"Heme": 1}, {}, kcat=1.0, km=0.4)],
   "Heme biosynthesis; porphyrias as in clinical genetics and toxicology texts")

_p("bilirubin", "Bilirubin handling",
   "Disposing of the heme left over from old red cells. A partial block is the harmless Gilbert syndrome; a complete "
   "one is Crigler-Najjar, where bilirubin reaches the brain.",
   ["Heme", "Biliverdin", "BilirubinU", "BilirubinC", "UDPGA", "CO"], ["Heme", "UDPGA"],
   [R("ho1", "Heme oxygenase", "HMOX1", {"Heme": 1}, {"Biliverdin": 1, "CO": 1}, kcat=1.2, km=0.3),
    R("bvr", "Biliverdin reductase", "BLVR", {"Biliverdin": 1}, {"BilirubinU": 1}, kcat=1.5, km=0.3),
    R("ugt", "UGT1A1 conjugation", "UGT1A1", {"BilirubinU": 1, "UDPGA": 1}, {"BilirubinC": 1}, kcat=1.9, km=0.3,
      deficiency="Gilbert syndrome when partial (harmless jaundice under stress) and Crigler-Najjar when severe (kernicterus risk)",
      drugs=[{"name": "Phenobarbital", "effect": "activates", "note": "induces this enzyme, which works in Crigler-Najjar type II"},
             {"name": "Atazanavir", "effect": "inhibits", "note": "causes benign jaundice by blocking this step"}]),
    R("excrete", "Conjugated bilirubin into bile", "MRP2", {"BilirubinC": 1}, {}, kcat=1.5, km=0.4,
      deficiency="Dubin-Johnson syndrome: conjugated bilirubin cannot be exported"),
    R("co_out", "Carbon monoxide exhaled", "Export", {"CO": 1}, {}, kcat=2.0, km=0.4)],
   "Bilirubin metabolism; syndromes as in clinical texts")

_p("pyrimidine", "Pyrimidine synthesis",
   "Building the other half of the DNA alphabet. It shares carbamoyl phosphate with the urea cycle, which is why a "
   "urea cycle block spills into orotic acid.",
   ["CP", "CarbAsp", "Orotate", "OMP", "UMP", "Asp", "PRPP"], ["Asp", "PRPP"],
   [R("cad_cps2", "CPS II", "CAD", {}, {"CP": 1}, kcat=0.5),
    R("cad_atc", "Aspartate transcarbamoylase", "CAD", {"CP": 1, "Asp": 1}, {"CarbAsp": 1}, kcat=1.2, km=0.3),
    R("dhodh", "Dihydroorotate dehydrogenase", "DHODH", {"CarbAsp": 1}, {"Orotate": 1}, kcat=1.2, km=0.3,
      drugs=[{"name": "Leflunomide", "effect": "inhibits", "note": "blocks this step to damp down immune cell proliferation"},
             {"name": "Teriflunomide", "effect": "inhibits", "note": "the same block, used in multiple sclerosis"}]),
    R("umps", "UMP synthase", "UMPS", {"Orotate": 1, "PRPP": 1}, {"OMP": 1}, kcat=1.2, km=0.3,
      deficiency="Hereditary orotic aciduria: orotic acid accumulates and megaloblastic anaemia follows; treated by giving uridine",
      drugs=[{"name": "Uridine", "effect": "activates", "note": "bypasses the block by supplying the product directly"}]),
    R("omp_dec", "OMP decarboxylase", "UMPS", {"OMP": 1}, {"UMP": 1}, kcat=1.5, km=0.3),
    R("ump_out", "UMP into nucleic acids", "Transfer", {"UMP": 1}, {}, kcat=1.5, km=0.4)],
   "De novo pyrimidine synthesis; orotic aciduria and drug targets as in clinical texts")

_p("glutathione", "Glutathione and oxidative defence",
   "The cell's main antioxidant, and the system that mops up the toxic metabolite of paracetamol. Running it down is "
   "how paracetamol overdose kills the liver, and refilling it is how the antidote works.",
   ["Cys", "GSH", "GSSG", "NADPH", "NADP", "ROS", "NAPQI", "Adduct"], ["Cys", "NADPH", "NADP", "ROS", "NAPQI"],
   [R("gcs", "Glutathione synthesis", "GCLC-GSS", {"Cys": 1}, {"GSH": 1}, kcat=1.0, km=0.3,
      deficiency="Glutathione synthetase deficiency: haemolysis and acidosis",
      drugs=[{"name": "N-acetylcysteine", "effect": "activates", "note": "supplies cysteine and is the antidote in paracetamol overdose"}]),
    R("gpx", "Glutathione peroxidase", "GPX", {"GSH": 2, "ROS": 1}, {"GSSG": 1}, kcat=0.9, km=0.3),
    R("gr", "Glutathione reductase", "GSR", {"GSSG": 1, "NADPH": 1}, {"GSH": 2, "NADP": 1}, kcat=2.4, km=0.3,
      deficiency="With G6PD deficiency upstream, NADPH runs short and red cells haemolyse under oxidative stress"),
    R("napqi", "Paracetamol metabolite detoxified by glutathione", "GST", {"NAPQI": 1, "GSH": 1}, {"Adduct": 1},
      kcat=1.5, km=0.3),
    R("adduct_out", "Conjugate excreted", "Export", {"Adduct": 1}, {}, kcat=1.5, km=0.4)],
   "Glutathione system; paracetamol toxicity as in clinical pharmacology texts")

_p("sphingolipid", "Sphingolipid breakdown (lysosomal)",
   "Lysosomal disposal of membrane lipids. Each enzyme has a storage disease named after it, and the treatments split "
   "into replacing the enzyme and reducing how much substrate is made.",
   ["GM2", "GM3", "Ceramide", "GlcCer", "Sphingosine", "SO4"], ["SO4"],
   [R("supply", "Membrane turnover supplying GM2", "Turnover", {}, {"GM2": 1}, kcat=0.3),
    R("hexa", "Beta-hexosaminidase A", "HEXA", {"GM2": 1}, {"GM3": 1}, kcat=1.2, km=0.3,
      deficiency="Tay-Sachs disease: GM2 ganglioside accumulates in neurons, with a cherry-red macula and regression"),
    R("neu", "Neuraminidase and galactosidase steps", "NEU-GLB1", {"GM3": 1}, {"Ceramide": 1}, kcat=1.5, km=0.3,
      deficiency="GM1 gangliosidosis when the galactosidase step fails"),
    R("gba", "Glucocerebrosidase", "GBA", {"GlcCer": 1}, {"Ceramide": 1}, kcat=1.2, km=0.3,
      deficiency="Gaucher disease: glucocerebroside accumulates in macrophages, enlarging liver and spleen and thinning bone",
      drugs=[{"name": "Imiglucerase", "effect": "activates", "note": "enzyme replacement therapy, supplying the missing enzyme"},
             {"name": "Miglustat", "effect": "inhibits", "note": "substrate reduction: blocks synthesis upstream so less accumulates"},
             {"name": "Ambroxol", "effect": "activates", "note": "chaperone that helps some mutant enzymes fold"}]),
    R("gcs", "Glucosylceramide synthesis", "UGCG", {"Ceramide": 1}, {"GlcCer": 1}, kcat=1.0, km=0.3,
      drugs=[{"name": "Miglustat", "effect": "inhibits", "note": "the step substrate reduction therapy blocks"},
             {"name": "Eliglustat", "effect": "inhibits", "note": "a more selective version of the same idea"}]),
    R("cdase", "Ceramidase", "ASAH1", {"Ceramide": 1}, {"Sphingosine": 1}, kcat=1.5, km=0.3,
      deficiency="Farber disease"),
    R("sph_out", "Sphingosine recycled", "Transfer", {"Sphingosine": 1}, {}, kcat=1.5, km=0.4)],
   "Sphingolipid degradation; storage diseases and their therapies as in clinical genetics texts")

_p("cholesterol", "Cholesterol synthesis",
   "Building cholesterol from acetyl-CoA. Its rate-limiting step is the target of statins, the most prescribed drugs "
   "in the world, and a block at the last step causes a malformation syndrome.",
   ["AcCoA", "HMGCoA", "Mevalonate", "IPP", "Squalene", "Lathosterol", "Desmosterol", "Cholesterol", "NADPH", "NADP"],
   ["AcCoA", "NADPH", "NADP"],
   [R("hmgcs", "HMG-CoA synthase", "HMGCS1", {"AcCoA": 2}, {"HMGCoA": 1}, kcat=0.36, km=0.3),
    R("hmgcr", "HMG-CoA reductase", "HMGCR", {"HMGCoA": 1, "NADPH": 1}, {"Mevalonate": 1, "NADP": 1}, kcat=3.0, km=0.3,
      regulators=[{"species": "Cholesterol", "effect": "inhibit", "k": 0.6}],
      deficiency="Mevalonate kinase deficiency downstream causes periodic fever syndromes",
      drugs=[{"name": "Statins", "effect": "inhibits", "note": "the rate-limiting step; blocking it lowers LDL cholesterol"}]),
    R("mvk", "Mevalonate kinase and decarboxylase", "MVK", {"Mevalonate": 1}, {"IPP": 1}, kcat=1.5, km=0.3,
      deficiency="Mevalonate kinase deficiency: recurrent fevers, from mild (hyper-IgD syndrome) to severe",
      drugs=[{"name": "Bisphosphonates", "effect": "inhibits", "note": "block the next step along, which is how they act on bone"}]),
    R("squal", "Squalene synthesis", "FDFT1", {"IPP": 1}, {"Squalene": 1}, kcat=1.5, km=0.3),
    R("lano", "Squalene to lathosterol", "SQLE-LSS", {"Squalene": 1}, {"Lathosterol": 1}, kcat=1.5, km=0.3),
    R("sc5d", "Lathosterol to desmosterol", "SC5D", {"Lathosterol": 1}, {"Desmosterol": 1}, kcat=0.9, km=0.3,
      deficiency="Lathosterolosis"),
    R("dhcr7", "7-dehydrocholesterol reductase", "DHCR7", {"Desmosterol": 1, "NADPH": 1}, {"Cholesterol": 1, "NADP": 1},
      kcat=2.4, km=0.3,
      deficiency="Smith-Lemli-Opitz syndrome: cholesterol cannot be finished, causing malformations and intellectual disability; treated by feeding cholesterol"),
    R("chol_use", "Cholesterol into membranes and hormones", "Transfer", {"Cholesterol": 1}, {}, kcat=1.0, km=0.4)],
   "Cholesterol biosynthesis; statin target and SLOS as in clinical texts")

_p("pyruvate", "Pyruvate at the crossroads",
   "Where sugar breakdown meets the TCA cycle, fat synthesis and lactate. A block here forces everything into lactate "
   "and is a common cause of congenital lactic acidosis.",
   ["Pyr", "AcCoA", "Lac", "OAA", "NAD", "NADH", "CoA", "CO2"], ["Pyr", "CO2"],
   [R("pdh", "Pyruvate dehydrogenase complex", "PDH", {"Pyr": 1, "CoA": 1, "NAD": 1}, {"AcCoA": 1, "NADH": 1, "CO2": 1},
      kcat=1.2, km=0.3,
      regulators=[{"species": "AcCoA", "effect": "inhibit", "k": 0.8}],
      deficiency="PDH complex deficiency: pyruvate cannot enter the TCA cycle, so lactate rises; the ketogenic diet bypasses it by supplying acetyl-CoA from fat",
      drugs=[{"name": "Thiamine", "effect": "activates", "note": "cofactor; some forms respond"},
             {"name": "Dichloroacetate", "effect": "activates", "note": "keeps the complex switched on"}]),
    R("ldh", "Lactate dehydrogenase", "LDH", {"Pyr": 1, "NADH": 1}, {"Lac": 1, "NAD": 1}, kcat=1.5, km=0.3,
      reversible=True),
    R("pc", "Pyruvate carboxylase", "PC", {"Pyr": 1, "CO2": 1}, {"OAA": 1}, kcat=0.8, km=0.3,
      deficiency="Pyruvate carboxylase deficiency: lactic acidosis and failure to refill the TCA cycle"),
    R("lac_out", "Lactate to the blood", "Export", {"Lac": 1}, {}, kcat=1.0, km=0.4),
    R("accoa_out", "Acetyl-CoA to the TCA cycle", "Transfer", {"AcCoA": 1}, {"CoA": 1}, kcat=1.5, km=0.4),
    R("oaa_out", "Oxaloacetate to the TCA cycle", "Transfer", {"OAA": 1}, {}, kcat=1.5, km=0.4),
    R("nadh_ox", "NADH reoxidation", "Respiration", {"NADH": 1}, {"NAD": 1}, kcat=1.5, km=0.3)],
   "Pyruvate metabolism; PDH deficiency as in clinical genetics texts")

_p("alcohol", "Alcohol metabolism",
   "Breaking down ethanol, and the two enzymes behind the flushing reaction and the disulfiram reaction. Methanol "
   "poisoning is treated by competing for the first enzyme.",
   ["EtOH", "Acetaldehyde", "Acetate", "NAD", "NADH", "MeOH", "Formaldehyde", "Formate"], ["EtOH", "MeOH"],
   [R("adh", "Alcohol dehydrogenase", "ADH", {"EtOH": 1, "NAD": 1}, {"Acetaldehyde": 1, "NADH": 1}, kcat=1.2, km=0.3,
      drugs=[{"name": "Fomepizole", "effect": "inhibits", "note": "blocks this enzyme in methanol or ethylene glycol poisoning so the toxic products never form"}]),
    R("aldh", "Aldehyde dehydrogenase 2", "ALDH2", {"Acetaldehyde": 1, "NAD": 1}, {"Acetate": 1, "NADH": 1},
      kcat=1.5, km=0.3,
      deficiency="ALDH2 variant, common in East Asia: acetaldehyde accumulates, causing the flushing reaction and raising oesophageal cancer risk",
      drugs=[{"name": "Disulfiram", "effect": "inhibits", "note": "blocks this step deliberately, so drinking causes an unpleasant reaction"}]),
    R("adh_meoh", "Alcohol dehydrogenase acting on methanol", "ADH", {"MeOH": 1, "NAD": 1},
      {"Formaldehyde": 1, "NADH": 1}, kcat=0.6, km=0.4,
      drugs=[{"name": "Fomepizole", "effect": "inhibits", "note": "the block that prevents blindness in methanol poisoning"},
             {"name": "Ethanol", "effect": "inhibits", "note": "the older antidote: competes for the same enzyme"}]),
    R("faldh", "Formaldehyde to formate", "ALDH", {"Formaldehyde": 1}, {"Formate": 1}, kcat=1.5, km=0.3),
    R("acetate_out", "Acetate onward", "Transfer", {"Acetate": 1}, {}, kcat=1.5, km=0.4),
    R("formate_out", "Formate cleared (slowly)", "Folate-dependent", {"Formate": 1}, {}, kcat=0.5, km=0.4),
    R("nadh_ox", "NADH reoxidation", "Respiration", {"NADH": 1}, {"NAD": 1}, kcat=1.5, km=0.3)],
   "Ethanol and methanol metabolism; antidotes as in clinical toxicology texts")

_p("polyol", "Polyol (sorbitol) pathway",
   "The overflow route that turns excess glucose into sorbitol. It runs hard in diabetes and is blamed for cataract "
   "and nerve damage, which is why aldose reductase inhibitors were developed.",
   ["Glc", "Sorbitol", "Fru", "NADPH", "NADP", "NAD", "NADH"], ["Glc", "NADPH", "NADP", "NAD", "NADH"],
   [R("akr1b1", "Aldose reductase", "AKR1B1", {"Glc": 1, "NADPH": 1}, {"Sorbitol": 1, "NADP": 1}, kcat=1.0, km=0.6,
      drugs=[{"name": "Epalrestat", "effect": "inhibits", "note": "aldose reductase inhibitor used for diabetic neuropathy in some countries"}]),
    R("sord", "Sorbitol dehydrogenase", "SORD", {"Sorbitol": 1, "NAD": 1}, {"Fru": 1, "NADH": 1}, kcat=1.0, km=0.4,
      deficiency="Sorbitol dehydrogenase deficiency: sorbitol accumulates, causing a hereditary neuropathy"),
    R("fru_out", "Fructose into glycolysis", "Transfer", {"Fru": 1}, {}, kcat=1.5, km=0.4)],
   "Polyol pathway; diabetic complications as in clinical texts")

_p("propionate", "Propionate to succinyl-CoA",
   "Where odd-chain fats and several amino acids join the TCA cycle. Two blocks here are the classic organic "
   "acidemias, and one of them responds to vitamin B12.",
   ["PropCoA", "MethylmalonylCoA", "SucCoA", "CO2", "ATP", "ADP", "B12"], ["CO2", "ATP", "ADP", "B12"],
   [R("supply", "Propionyl-CoA from amino acids and odd-chain fat", "Upstream", {}, {"PropCoA": 1}, kcat=0.4),
    R("pcc", "Propionyl-CoA carboxylase", "PCC", {"PropCoA": 1, "CO2": 1, "ATP": 1},
      {"MethylmalonylCoA": 1, "ADP": 1}, kcat=1.2, km=0.3,
      deficiency="Propionic acidemia: propionyl-CoA accumulates, causing acidosis, hyperammonaemia and cardiomyopathy; biotin is its cofactor",
      drugs=[{"name": "Biotin", "effect": "activates", "note": "cofactor; helps in some multiple carboxylase forms"}]),
    R("mut", "Methylmalonyl-CoA mutase", "MMUT", {"MethylmalonylCoA": 1, "B12": 1}, {"SucCoA": 1, "B12": 1},
      kcat=1.2, km=0.3,
      deficiency="Methylmalonic acidemia: methylmalonic acid accumulates; some forms are B12-responsive",
      drugs=[{"name": "Hydroxocobalamin (B12)", "effect": "activates", "note": "restores the cofactor in responsive forms"}]),
    R("suc_out", "Succinyl-CoA to the TCA cycle", "Transfer", {"SucCoA": 1}, {}, kcat=1.5, km=0.4)],
   "Propionate metabolism; organic acidemias as in clinical genetics texts")

_p("catecholamine", "Catecholamine synthesis",
   "Making dopamine, noradrenaline and adrenaline from tyrosine. It is the pathway Parkinson's treatment works on, "
   "and where PKU's tyrosine shortage bites.",
   ["Tyr", "LDOPA", "Dopamine", "NE", "Epi", "BH4", "HVA"], ["Tyr", "BH4"],
   [R("th", "Tyrosine hydroxylase", "TH", {"Tyr": 1, "BH4": 1}, {"LDOPA": 1, "BH4": 1}, kcat=0.8, km=0.4,
      deficiency="Tyrosine hydroxylase deficiency: dopamine cannot be made, causing a dystonia that responds to L-DOPA",
      drugs=[{"name": "Metyrosine", "effect": "inhibits", "note": "deliberately blocks catecholamine synthesis before phaeochromocytoma surgery"}]),
    R("aadc", "Aromatic amino acid decarboxylase", "DDC", {"LDOPA": 1}, {"Dopamine": 1}, kcat=1.5, km=0.3,
      deficiency="AADC deficiency: severe movement disorder from infancy; now treatable by gene therapy",
      drugs=[{"name": "Carbidopa", "effect": "inhibits", "note": "blocks this step outside the brain so more L-DOPA reaches it"}]),
    R("dbh", "Dopamine beta-hydroxylase", "DBH", {"Dopamine": 1}, {"NE": 1}, kcat=1.0, km=0.3,
      deficiency="DBH deficiency: no noradrenaline, causing severe orthostatic hypotension"),
    R("pnmt", "Phenylethanolamine N-methyltransferase", "PNMT", {"NE": 1}, {"Epi": 1}, kcat=1.0, km=0.3),
    R("mao", "Monoamine oxidase and COMT", "MAO-COMT", {"Dopamine": 1}, {"HVA": 1}, kcat=1.0, km=0.3,
      drugs=[{"name": "Selegiline", "effect": "inhibits", "note": "slows dopamine breakdown in Parkinson's disease"},
             {"name": "Entacapone", "effect": "inhibits", "note": "blocks the COMT side of the same disposal route"}]),
    R("ne_use", "Noradrenaline released and cleared", "Transfer", {"NE": 1}, {}, kcat=1.2, km=0.4),
    R("epi_use", "Adrenaline released and cleared", "Transfer", {"Epi": 1}, {}, kcat=1.2, km=0.4),
    R("hva_out", "Homovanillic acid excreted", "Export", {"HVA": 1}, {}, kcat=1.5, km=0.4)],
   "Catecholamine synthesis and breakdown; drugs as in clinical pharmacology texts")


def _apply_tuning():
    """Rate constants balanced by tools/balance.py so each pathway reaches a steady state."""
    import json
    from pathlib import Path
    f = Path(__file__).resolve().parent / "rate_tuning.json"
    if not f.exists():
        return
    tuned = json.loads(f.read_text())
    for pid, path in PATHWAYS.items():
        for r in path["reactions"]:
            if pid in tuned and r["id"] in tuned[pid]:
                r["kcat"] = tuned[pid][r["id"]]


_apply_tuning()


def _complex_key(d):
    return tuple(sorted(d.items()))


def stoichiometry(path):
    sp = path["species"]
    idx = {s: i for i, s in enumerate(sp)}
    cols = []
    for r in path["reactions"]:
        v = np.zeros(len(sp))
        for s, k in r["substrates"].items():
            v[idx[s]] -= k
        for s, k in r["products"].items():
            v[idx[s]] += k
        cols.append(v)
    return np.stack(cols, axis=1) if cols else np.zeros((len(sp), 0))


def conservation_laws(N, species, clamped=(), max_support=None, max_coef=3):
    """Small integer conserved pools: y >= 0 with y @ N = 0, over species that are not held fixed."""
    from itertools import combinations, product
    free = [i for i, sp in enumerate(species) if sp not in clamped]
    if max_support is None:
        max_support = 4 if len(free) <= 30 else 3
    rows = {i: N[i] for i in free}
    laws, supports = [], []
    for size in range(2, max_support + 1):
        for combo in combinations(free, size):
            if any(set(prev) <= set(combo) for prev in supports):
                continue
            base = np.stack([rows[i] for i in combo])
            for coefs in product(range(1, max_coef + 1), repeat=size):
                if np.gcd.reduce(coefs) != 1:
                    continue
                if np.abs(np.asarray(coefs) @ base).max() < 1e-9:
                    laws.append([(species[i], int(c)) for i, c in zip(combo, coefs)])
                    supports.append(combo)
                    break
            if len(laws) >= 8:
                return laws
    return laws


def deficiency(path):
    complexes, links = [], []
    for r in path["reactions"]:
        a, b = _complex_key(r["substrates"]), _complex_key(r["products"])
        for c in (a, b):
            if c not in complexes:
                complexes.append(c)
        links.append((complexes.index(a), complexes.index(b)))
    n = len(complexes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    l = len({find(i) for i in range(n)})
    N = stoichiometry(path)
    s = int(np.linalg.matrix_rank(N)) if N.size else 0
    reach = {i: set() for i in range(n)}
    for a, b in links:
        reach[a].add(b)
        if any(r["reversible"] for r in path["reactions"]):
            pass
    for r, (a, b) in zip(path["reactions"], links):
        if r["reversible"]:
            reach[b].add(a)
    changed = True
    while changed:
        changed = False
        for i in list(reach):
            for j in list(reach[i]):
                new = reach[j] - reach[i]
                if new:
                    reach[i] |= new
                    changed = True
    weakly_reversible = all(i in reach[j] for i in range(n) for j in reach[i])
    return {"complexes": n, "linkage_classes": l, "rank": s, "deficiency": n - l - s,
            "weakly_reversible": bool(weakly_reversible)}


def influence_signs(path):
    """Sign of each species' effect on each other species, from substrate use, product formation and regulation."""
    sp = path["species"]
    idx = {s: i for i, s in enumerate(sp)}
    edges = {}
    for r in path["reactions"]:
        actors = [(s, +1) for s in r["substrates"]] + [(g["species"], +1 if g["effect"] == "activate" else -1) for g in r["regulators"]]
        targets = [(s, -1) for s in r["substrates"]] + [(s, +1) for s in r["products"]]
        for a, sa in actors:
            for b, sb in targets:
                if a == b:
                    continue
                edges.setdefault((idx[a], idx[b]), set()).add(sa * sb)
        if r["reversible"]:
            for a in r["products"]:
                for b, sb in [(s, +1) for s in r["substrates"]] + [(s, -1) for s in r["products"]]:
                    if a == b:
                        continue
                    edges.setdefault((idx[a], idx[b]), set()).add(sb)
    return edges


def sign_consistency(path):
    """Is the influence graph sign-consistent (no negative undirected cycle)? Monotone if it is."""
    edges = influence_signs(path)
    n = len(path["species"])
    parent, par = list(range(n)), [0] * n

    def find(x):
        p = 0
        while parent[x] != x:
            p ^= par[x]
            x = parent[x]
        return x, p

    conflicts, ambiguous = [], []
    for (a, b), signs in edges.items():
        if len(signs) > 1:
            ambiguous.append((path["species"][a], path["species"][b]))
            continue
        sgn = next(iter(signs))
        ra, pa = find(a)
        rb, pb = find(b)
        need = 0 if sgn > 0 else 1
        if ra == rb:
            if (pa ^ pb) != need:
                conflicts.append((path["species"][a], path["species"][b]))
        else:
            parent[ra] = rb
            par[ra] = pa ^ pb ^ need
    return {"consistent": not conflicts and not ambiguous, "conflicts": conflicts[:8], "ambiguous": ambiguous[:8],
            "edges": len(edges)}


def analyse(path):
    N = stoichiometry(path)
    d = deficiency(path)
    sc = sign_consistency(path)
    laws = conservation_laws(N, path["species"], path["clamped"])
    notes = []
    if d["deficiency"] == 0 and d["weakly_reversible"]:
        notes.append("Deficiency zero and weakly reversible: by the Deficiency Zero Theorem, with mass-action kinetics this "
                     "network has exactly one positive steady state in each conserved class, and it is stable, for every "
                     "choice of rate constants.")
    elif d["deficiency"] == 0:
        notes.append("Deficiency zero but not weakly reversible. For the closed network with mass-action kinetics, the "
                     "theorem rules out a positive steady state whatever the rate constants. That does not contradict the "
                     "simulation below, which holds some pools fixed and uses saturating enzyme kinetics, so it is a "
                     "different, open system.")
    elif d["deficiency"] == 1:
        notes.append("Deficiency one: the Deficiency One Theorem may apply, and absolute concentration robustness becomes "
                     "possible, where one species holds the same steady-state level no matter how much total material there is.")
    else:
        notes.append(f"Deficiency {d['deficiency']}: too high for the deficiency theorems, so steady-state behaviour depends "
                     "on the rate constants and has to be studied numerically.")
    if sc["consistent"]:
        notes.append("The influence graph is sign-consistent, so this network is monotone: it cannot sustain oscillations, "
                     "and almost every trajectory settles. This is the same structure the bench's sign-consistent arm enforces.")
    elif sc["conflicts"]:
        notes.append(f"The influence graph has {len(sc['conflicts'])} sign conflict(s), so the network contains negative "
                     "feedback cycles. Oscillation and multiple stable states are possible.")
    else:
        notes.append("Some species affect each other in both directions depending on the reaction, so the network is not "
                     "sign-consistent and no monotonicity guarantee applies.")
    if laws:
        pools = "; ".join(" + ".join((f"{c} {sp}" if c > 1 else sp) for sp, c in law) for law in laws[:4])
        notes.append(f"{len(laws)} conserved pool(s), quantities the reactions move around but never create or destroy: {pools}.")
    return {"species": path["species"], "reactions": [{"id": r["id"], "name": r["name"], "enzyme": r["enzyme"],
                                                       "substrates": r["substrates"], "products": r["products"],
                                                       "reversible": r["reversible"], "regulators": r["regulators"],
                                                       "deficiency": r.get("deficiency"), "drugs": r.get("drugs", [])}
                                                      for r in path["reactions"]],
            "stoichiometry": N.astype(int).tolist(), "conservation": laws, "structure": d, "signs": sc,
            "notes": notes, "clamped": path["clamped"], "name": path["name"], "description": path["description"],
            "reference": path["reference"]}


def rates(path, c, idx, enzyme_scale=None):
    """Reaction rates. Each reaction picks a rate law:
    mm           kcat * product of x^k / (Km^k + x^k), saturating enzyme kinetics (the default)
    mass_action  kcat * product of x^k, for encounters between molecules, cells or viruses
    hill         kcat * product of x^n / (K^n + x^n), cooperative switch-like response
    A reaction with no substrates runs at a constant rate kcat (a supply or a birth rate).
    Regulators multiply the rate: inhibitors by K^n / (K^n + x^n), activators by 0.3 + 0.7 x^n / (K^n + x^n).
    """
    v = np.zeros((c.shape[0], len(path["reactions"])))
    for j, r in enumerate(path["reactions"]):
        law = r.get("law", "mm")
        f = np.full(c.shape[0], float(r["kcat"]))
        for s, k in r["substrates"].items():
            x = np.maximum(c[:, idx[s]], 0.0)
            if law == "mass_action":
                f = f * x ** k
            elif law == "hill":
                n = r.get("n", 2)
                f = f * x ** n / (r["km"] ** n + x ** n)
            else:
                f = f * (x ** k) / (r["km"] ** k + x ** k)
        for g in r["regulators"]:
            x = np.maximum(c[:, idx[g["species"]]], 0.0)
            n = g.get("n", 1)
            kk = g["k"] ** n
            f = f * (kk / (kk + x ** n) if g["effect"] == "inhibit" else 0.3 + 0.7 * x ** n / (kk + x ** n))
        if r["reversible"]:
            b = np.full(c.shape[0], r["kcat"] * 0.4)
            for s, k in r["products"].items():
                x = np.maximum(c[:, idx[s]], 0.0)
                b = b * (x ** k if law == "mass_action" else (x ** k) / (r["km"] ** k + x ** k))
            f = f - b
        if enzyme_scale is not None:
            f = f * enzyme_scale[j]
        v[:, j] = f
    return v


COFACTORS = ("ATP", "NAD", "NADP", "CoA", "OAA", "GDP", "Pi", "Glc", "AcCoA", "G6P", "Pyr", "Orn", "UDPGlc",
             "PRPP", "Asp", "Ado", "Hx", "Phe", "Gal", "NH3", "CO2")


def initial_state(path, level=1.0):
    if path.get("initial"):
        c = {s: 0.0 for s in path["species"]}
        c.update({k: float(v) for k, v in path["initial"].items() if k in c})
        return c
    c = {s: 0.05 for s in path["species"]}
    for s in path["clamped"]:
        c[s] = level
    big = 1.5 if path.get("id") == "all" else 0.6
    for s in COFACTORS:
        if s in c:
            c[s] = max(c[s], big)
    for s in ("ADP", "AMP", "NADH", "NADPH", "GTP"):
        if s in c:
            c[s] = 0.15
    return c


def simulate_pathway(path, steps=400, dt=0.05, inputs=None, enzyme_scale=None, c0=None, record=True):
    sp = path["species"]
    idx = {s: i for i, s in enumerate(sp)}
    c = np.array([[(c0 or initial_state(path))[s] for s in sp]], dtype=float)
    N = stoichiometry(path)
    clamp = np.array([s in path["clamped"] for s in sp])
    trace = []
    for t in range(steps):
        v = rates(path, c, idx, enzyme_scale)
        dc = (N @ v[0])[None, :]
        if inputs:
            for s, amt in inputs.items():
                dc[0, idx[s]] += amt
        c = np.clip(c + dt * dc, 0.0, None)
        c[:, clamp] = np.array([[(c0 or initial_state(path))[s] for s in sp]])[:, clamp]
        if record and t % max(1, steps // 200) == 0:
            trace.append(np.round(c[0], 4).tolist())
    flux = rates(path, c, idx, enzyme_scale)[0]
    residual = float(np.abs((N @ flux)[~clamp]).max())
    return {"species": sp, "trace": trace, "final": np.round(c[0], 4).tolist(), "residual": round(residual, 5),
            "flux": {r["id"]: round(float(f), 4) for r, f in zip(path["reactions"], flux)},
            "steady": bool(residual < 5e-3)}


def backbone_sign_consistency(path):
    """Is the pathway monotone once the allosteric regulation is removed?"""
    bare = dict(path)
    bare["reactions"] = [dict(r, regulators=[]) for r in path["reactions"]]
    return sign_consistency(bare)


def acr_scan(path, factors=(0.7, 1.0, 1.4), steps=2500, tol=0.02):
    """Species whose steady level barely moves when every free pool is scaled: candidates for
    absolute concentration robustness."""
    sp = path["species"]
    finals = []
    for f in factors:
        c0 = initial_state(path)
        c0 = {k: (v * f if k not in path["clamped"] else v) for k, v in c0.items()}
        finals.append(np.array(simulate_pathway(path, steps=steps, c0=c0, record=False)["final"]))
    F = np.stack(finals)
    lo, hi = F.min(axis=0), F.max(axis=0)
    spread = (hi - lo) / np.maximum(hi, 1e-6)
    robust, sensitive = [], []
    for i, s in enumerate(sp):
        if s in path["clamped"] or hi[i] < 1e-3:
            continue
        (robust if spread[i] < tol else sensitive).append(
            {"species": s, "spread": round(float(spread[i]), 4), "level": round(float(F[1, i]), 4)})
    sensitive.sort(key=lambda d: -d["spread"])
    return {"robust": robust, "sensitive": sensitive[:6], "factors": list(factors)}


def steady_states(path, tries=8, steps=2500, seed=2, merge=0.05):
    """Distinct steady states from different starting points *within the same conserved pools*, so that
    differences mean multistability rather than different amounts of material."""
    rng = np.random.default_rng(seed)
    N = stoichiometry(path)
    pools = conservation_laws(N, path["species"], path["clamped"])
    base = initial_state(path)
    ends = []
    for k in range(tries):
        c0 = {s: (v if s in path["clamped"] else float(np.clip(v * rng.uniform(0.2, 2.5), 1e-6, None)))
              for s, v in base.items()}
        for law in pools:
            want = sum(c * base[sp] for sp, c in law)
            have = sum(c * c0[sp] for sp, c in law)
            if have > 1e-9:
                for sp, _ in law:
                    c0[sp] *= want / have
        out = simulate_pathway(path, steps=steps, c0=c0, record=False)
        ends.append((np.array(out["final"]), out["residual"]))
    distinct = []
    unsettled = 0
    for e, res in ends:
        if res > 5e-3:
            unsettled += 1
            continue
        if not any(np.abs(e - d).max() < merge for d in distinct):
            distinct.append(e)
    return {"tries": tries, "distinct": len(distinct), "unsettled": unsettled}


def control_coefficients(path, target=None, steps=2500, delta=0.15):
    """Flux control coefficients: the fractional change in the pathway's output flux per fractional change
    in each enzyme. This is the measured drug-target ranking."""
    run = simulate_pathway(path, steps=steps, record=False)
    flux = run["flux"]
    if target is None:
        exits = [r["id"] for r in path["reactions"] if not r["products"]]
        target = max(exits, key=lambda k: flux[k]) if exits else max(flux, key=lambda k: flux[k])
    base = flux[target]
    rows = []
    for j, r in enumerate(path["reactions"]):
        out = []
        for f in (1 - delta, 1 + delta):
            scale = [f if k == j else 1.0 for k in range(len(path["reactions"]))]
            out.append(simulate_pathway(path, steps=steps, enzyme_scale=scale, record=False)["flux"][target])
        c = ((out[1] - out[0]) / max(abs(base), 1e-6)) / (2 * delta)
        rows.append({"reaction": r["id"], "enzyme": r["enzyme"], "control": round(float(c), 3)})
    rows.sort(key=lambda d: -abs(d["control"]))
    name = next(r["enzyme"] for r in path["reactions"] if r["id"] == target)
    return {"target": target, "target_name": name, "base": round(float(base), 4), "coefficients": rows,
            "sum": round(float(sum(x["control"] for x in rows)), 3)}


def deeper(path):
    bb = backbone_sign_consistency(path)
    acr = acr_scan(path)
    ss = steady_states(path)
    cc = control_coefficients(path)
    notes = []
    full = sign_consistency(path)
    if bb["consistent"] and not full["consistent"]:
        notes.append("Strip the allosteric regulation and the pathway becomes sign-consistent, so its backbone is "
                     "monotone and the regulation is what breaks it. That is the weaker property real pathways share: "
                     "a monotone skeleton plus a handful of feedback loops. The bench's feedback arm enforces exactly this.")
    elif bb["consistent"]:
        notes.append("The pathway is sign-consistent with or without its regulation, so it is monotone as it stands.")
    else:
        notes.append("Even with the regulation removed, the reaction wiring itself is not sign-consistent, so this "
                     "pathway is not monotone at any level.")
    if acr["robust"]:
        notes.append("Species that held their level when every free pool was scaled by 0.7 and 1.4: "
                     + ", ".join(d["species"] for d in acr["robust"][:8])
                     + ". These are candidates for absolute concentration robustness, where the steady level does not "
                       "depend on how much material there is. This is a numerical test, not a proof.")
    else:
        notes.append("No species held its level when the pools were scaled, so nothing here looks concentration-robust.")
    if ss["distinct"] > 1:
        notes.append(f"From {ss['tries']} different starting points the pathway settled into {ss['distinct']} distinct "
                     "steady states, so it is multistable: where it starts decides where it ends.")
    elif ss["unsettled"]:
        notes.append(f"{ss['unsettled']} of {ss['tries']} starting points had not settled by the end of the run.")
    else:
        notes.append("Every starting point settled to the same steady state, so no multistability showed up here.")
    if ss["distinct"] > 1:
        notes.append("Because the pathway is multistable, treat the control coefficients below with care: a small "
                     "change in an enzyme can tip it into a different steady state, which shows up as a huge coefficient.")
    top = cc["coefficients"][0] if cc["coefficients"] else None
    if top:
        notes.append(f"Control over the pathway's output flux ({cc['target_name']}) sits mostly with {top['enzyme']} "
                     f"(flux control coefficient {top['control']}; the coefficients sum to {cc['sum']}, and for a "
                     "pathway at steady state the sum should be close to 1). The enzyme with most control is the one "
                     "worth inhibiting, and the ones near zero are the ones a drug would barely move. These are "
                     "measured here, and depend on the illustrative rate constants.")
    return {"backbone": bb, "acr": acr, "steady_states": ss, "control": cc, "notes": notes}


def graph(path):
    """Bipartite graph of species and reactions, ready for drawing or for loading into a graph database."""
    nodes = [{"id": f"s:{s}", "kind": "species", "label": s, "clamped": s in path["clamped"]} for s in path["species"]]
    nodes += [{"id": f"r:{r['id']}", "kind": "reaction", "label": r["enzyme"], "name": r["name"],
               "deficiency": r.get("deficiency"), "drugs": r.get("drugs", []), "reversible": r["reversible"]}
              for r in path["reactions"]]
    edges = []
    for r in path["reactions"]:
        for sp, k in r["substrates"].items():
            edges.append({"from": f"s:{sp}", "to": f"r:{r['id']}", "type": "SUBSTRATE_OF", "stoich": k})
        for sp, k in r["products"].items():
            edges.append({"from": f"r:{r['id']}", "to": f"s:{sp}", "type": "PRODUCT_OF", "stoich": k})
        for g in r["regulators"]:
            edges.append({"from": f"s:{g['species']}", "to": f"r:{r['id']}", "type": "REGULATES", "effect": g["effect"]})
    return {"id": path["id"], "name": path["name"], "nodes": nodes, "edges": edges}


def _cy(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"


def cypher(path):
    """Cypher statements that build this pathway in a graph database such as Neo4j."""
    L = [f"// {path['name']} — generated by Fate Memory Lab",
         f"MERGE (p:Pathway {{id: {_cy(path['id'])}}}) SET p.name = {_cy(path['name'])}, "
         f"p.description = {_cy(path['description'])}, p.reference = {_cy(path['reference'])};"]
    for sp in path["species"]:
        L.append(f"MERGE (m:Metabolite {{name: {_cy(sp)}, pathway: {_cy(path['id'])}}}) "
                 f"SET m.held_fixed = {_cy(sp in path['clamped'])} "
                 f"WITH m MATCH (p:Pathway {{id: {_cy(path['id'])}}}) MERGE (m)-[:IN_PATHWAY]->(p);")
        L.append(f"MERGE (c:Compound {{name: {_cy(sp)}}}) WITH c "
                 f"MATCH (m:Metabolite {{name: {_cy(sp)}, pathway: {_cy(path['id'])}}}) MERGE (m)-[:IS]->(c);")
    for r in path["reactions"]:
        L.append(f"MERGE (x:Reaction {{id: {_cy(r['id'])}, pathway: {_cy(path['id'])}}}) "
                 f"SET x.name = {_cy(r['name'])}, x.enzyme = {_cy(r['enzyme'])}, x.kcat = {_cy(r['kcat'])}, "
                 f"x.km = {_cy(r['km'])}, x.reversible = {_cy(r['reversible'])}, x.deficiency = {_cy(r.get('deficiency'))} "
                 f"WITH x MATCH (p:Pathway {{id: {_cy(path['id'])}}}) MERGE (x)-[:IN_PATHWAY]->(p);")
        L.append(f"MERGE (e:Enzyme {{name: {_cy(r['enzyme'])}}}) WITH e "
                 f"MATCH (x:Reaction {{id: {_cy(r['id'])}, pathway: {_cy(path['id'])}}}) MERGE (e)-[:CATALYSES]->(x);")
        for sp, k in r["substrates"].items():
            L.append(f"MATCH (m:Metabolite {{name: {_cy(sp)}, pathway: {_cy(path['id'])}}}), "
                     f"(x:Reaction {{id: {_cy(r['id'])}, pathway: {_cy(path['id'])}}}) "
                     f"MERGE (m)-[:SUBSTRATE_OF {{stoichiometry: {_cy(k)}}}]->(x);")
        for sp, k in r["products"].items():
            L.append(f"MATCH (x:Reaction {{id: {_cy(r['id'])}, pathway: {_cy(path['id'])}}}), "
                     f"(m:Metabolite {{name: {_cy(sp)}, pathway: {_cy(path['id'])}}}) "
                     f"MERGE (x)-[:PRODUCES {{stoichiometry: {_cy(k)}}}]->(m);")
        for g in r["regulators"]:
            L.append(f"MATCH (m:Metabolite {{name: {_cy(g['species'])}, pathway: {_cy(path['id'])}}}), "
                     f"(x:Reaction {{id: {_cy(r['id'])}, pathway: {_cy(path['id'])}}}) "
                     f"MERGE (m)-[:REGULATES {{effect: {_cy(g['effect'])}, constant: {_cy(g['k'])}}}]->(x);")
        for d in r.get("drugs", []):
            L.append(f"MERGE (d:Drug {{name: {_cy(d['name'])}}}) WITH d "
                     f"MATCH (x:Reaction {{id: {_cy(r['id'])}, pathway: {_cy(path['id'])}}}) "
                     f"MERGE (d)-[:ACTS_ON {{effect: {_cy(d['effect'])}, note: {_cy(d.get('note'))}}}]->(x);")
        if r.get("deficiency"):
            L.append(f"MERGE (c:Condition {{name: {_cy(r['deficiency'].split(':')[0])}}}) "
                     f"SET c.description = {_cy(r['deficiency'])} WITH c "
                     f"MATCH (e:Enzyme {{name: {_cy(r['enzyme'])}}}) MERGE (c)-[:CAUSED_BY_LOSS_OF]->(e);")
    return "\n".join(L)


def what_if(path, node_id, levels=(0.5, 0.2, 0.0), steps=3000, boost=3.0, tol=0.05, attribute=True):
    """What happens if one thing changes: an enzyme loses activity, or a metabolite is supplied in excess."""
    sp = path["species"]
    base = simulate_pathway(path, steps=steps, record=False)
    kind, key = node_id.split(":", 1)
    out = {"node": node_id, "kind": kind, "label": key, "baseline_flux": base["flux"], "cases": []}
    if kind == "r":
        r = next((x for x in path["reactions"] if x["id"] == key), None)
        out["in_pathways"] = path.get("origin_reactions", {}).get(key, [path["id"]])
        if r is None:
            return None
        out["label"] = r["enzyme"]
        out["name"] = r["name"]
        out["deficiency"] = r.get("deficiency")
        out["drugs"] = r.get("drugs", [])
        for lv in levels:
            scale = [lv if x["id"] == key else 1.0 for x in path["reactions"]]
            run = simulate_pathway(path, steps=steps, enzyme_scale=scale, record=False)
            case = _compare(sp, base, run, f"{int(lv * 100)}% activity", tol)
            if attribute and path.get("origin_species"):
                case["pathways"] = affected_pathways(path, case["changes"])
            out["cases"].append(case)
    else:
        if key not in sp:
            return None
        out["name"] = key
        for f in (boost, 1 / boost):
            c0 = initial_state(path)
            c0[key] = c0[key] * f
            held = dict(path)
            held["clamped"] = sorted(set(path["clamped"]) | {key})
            run = simulate_pathway(held, steps=steps, c0=c0, record=False)
            case = _compare(sp, base, run, f"{key} held {'high' if f > 1 else 'low'} ({round(f, 2)}x)", tol)
            if attribute and path.get("origin_species"):
                case["pathways"] = affected_pathways(path, case["changes"])
            out["cases"].append(case)
    return out


def _compare(sp, base, run, label, tol):
    changes = []
    for i, s in enumerate(sp):
        b, k = base["final"][i], run["final"][i]
        if abs(k - b) > tol and max(b, k) > 1e-3:
            changes.append({"species": s, "before": round(float(b), 3), "after": round(float(k), 3),
                            "fold": round(float((k + 1e-6) / (b + 1e-6)), 2)})
    changes.sort(key=lambda d: -abs(np.log(max(d["fold"], 1e-6))))
    fl = []
    for rid, b in base["flux"].items():
        k = run["flux"][rid]
        if abs(k - b) > 0.02:
            fl.append({"reaction": rid, "before": round(float(b), 3), "after": round(float(k), 3)})
    fl.sort(key=lambda d: -abs(d["after"] - d["before"]))
    return {"label": label, "changes": changes[:8], "fluxes": fl[:6], "settled": run["steady"]}


def _merged_tuning():
    import json
    from pathlib import Path
    f = Path(__file__).resolve().parent / "rate_tuning.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text()).get("all", {})


def merged(pids=None):
    """All pathways joined into one network through the metabolites they share."""
    pids = list(pids or PATHWAYS)
    species, clamped_votes, holders, producers = [], {}, {}, set()
    reactions, seen = [], {}
    origin_r, origin_s = {}, {}
    for pid in pids:
        p = PATHWAYS[pid]
        for sp in p["species"]:
            if sp not in species:
                species.append(sp)
            holders.setdefault(sp, []).append(pid)
            clamped_votes.setdefault(sp, []).append(sp in p["clamped"])
            origin_s.setdefault(sp, []).append(pid)
        for r in p["reactions"]:
            key = (tuple(sorted(r["substrates"].items())), tuple(sorted(r["products"].items())), r["enzyme"])
            if key in seen:
                origin_r[seen[key]].append(pid)
                continue
            rid = f"{pid}_{r['id']}"
            seen[key] = rid
            origin_r[rid] = [pid]
            reactions.append(dict(r, id=rid))
            producers.update(r["products"])
    environment = {"Glc", "Pi", "CO2", "H", "AMP", "GDP", "Asp", "PRPP", "Ado", "O2"}
    clamped = [sp for sp in species if (all(clamped_votes[sp]) and sp not in producers) or sp in environment]
    tuned = _merged_tuning()
    for r in reactions:
        if r["id"] in tuned:
            r["kcat"] = tuned[r["id"]]
    return {"id": "all", "name": "Whole metabolism (all pathways joined)",
            "description": "Every pathway in the app merged into one network through the metabolites they share, so a "
                           "change in one pathway can be followed into the others. Reactions that appear in more than "
                           "one pathway are kept once.",
            "species": species, "clamped": clamped, "reactions": reactions,
            "reference": "Union of the individual pathways listed in this app",
            "origin_reactions": origin_r, "origin_species": origin_s, "members": pids}


def affected_pathways(path, changes):
    """Which source pathways contain the metabolites that moved."""
    org = path.get("origin_species", {})
    out = {}
    for c in changes:
        for pid in org.get(c["species"], []):
            out.setdefault(pid, []).append(c["species"])
    return {k: sorted(set(v)) for k, v in out.items()}


def _levels(run, species):
    return {s: v for s, v in zip(species, run["final"])}


def _neighbourhood(path, rid, hops=2, cap=24):
    """Reactions within a few steps of one reaction, through the metabolites they share."""
    by_species = {}
    for r in path["reactions"]:
        for sp in list(r["substrates"]) + list(r["products"]):
            by_species.setdefault(sp, []).append(r["id"])
    start = next((r for r in path["reactions"] if r["id"] == rid), None)
    if start is None:
        return set()
    frontier, seen = {rid}, {rid}
    for _ in range(hops):
        nxt = set()
        for x in frontier:
            rx = next(r for r in path["reactions"] if r["id"] == x)
            for sp in list(rx["substrates"]) + list(rx["products"]):
                for y in by_species.get(sp, []):
                    if y not in seen:
                        nxt.add(y)
        seen |= nxt
        frontier = nxt
        if len(seen) >= cap:
            break
    return set(list(seen)[:cap])


def strategies(path, reaction_id, residual=0.05, steps=3000, top=8, combos=True):
    """Given an enzyme that has failed, try the standard therapeutic moves in the model and rank them.

    The moves mirror how metabolic disease is actually treated: restore the enzyme, reduce what flows in,
    block a step further up, drain the toxic metabolite, supply what is missing downstream, or push an
    alternative route.
    """
    sp = path["species"]
    rxn = next((r for r in path["reactions"] if r["id"] == reaction_id), None)
    if rxn is None:
        return None
    big = len(path["reactions"]) > 30
    if big:
        steps = min(steps, 1200)
        combos = False
    ids = [r["id"] for r in path["reactions"]]
    j = ids.index(reaction_id)
    healthy = simulate_pathway(path, steps=steps, record=False)
    sick_scale = [residual if i == j else 1.0 for i in range(len(ids))]
    sick = simulate_pathway(path, steps=steps, enzyme_scale=sick_scale, record=False)
    h, d = _levels(healthy, sp), _levels(sick, sp)
    toxin, tox_fold = None, 1.0
    for s in sp:
        if s in path["clamped"]:
            continue
        fold = (d[s] + 1e-4) / (h[s] + 1e-4)
        if fold > tox_fold and d[s] > 0.05:
            toxin, tox_fold = s, fold
    exits = [r["id"] for r in path["reactions"] if not r["products"]]
    out_flux = max(exits, key=lambda k: healthy["flux"][k]) if exits else max(healthy["flux"], key=lambda k: healthy["flux"][k])
    h_out, d_out = healthy["flux"][out_flux], sick["flux"][out_flux]

    def score(run):
        lv = _levels(run, sp)
        s_tox = 1.0
        if toxin:
            span = max(np.log((d[toxin] + 1e-4) / (h[toxin] + 1e-4)), 1e-6)
            s_tox = 1 - min(max(np.log((lv[toxin] + 1e-4) / (h[toxin] + 1e-4)), 0) / span, 1)
        gap = max(h_out - d_out, 1e-6)
        s_flux = min(max((run["flux"][out_flux] - d_out) / gap, 0), 1.2)
        return round(float(0.85 * s_tox + 0.15 * s_flux), 3), round(float(lv[toxin]) if toxin else 0.0, 3), \
            round(float(run["flux"][out_flux]), 3)

    def run_with(scale=None, clamp=None, c0=None, extra_exit=None):
        p2 = dict(path)
        p2["reactions"] = list(path["reactions"])
        if extra_exit:
            p2["reactions"] = p2["reactions"] + [R(f"scav_{extra_exit}", f"Scavenger drug removing {extra_exit}",
                                                   "Scavenger", {extra_exit: 1}, {}, kcat=1.5, km=0.3)]
            scale = (scale or [1.0] * len(ids)) + [1.0]
        if clamp:
            p2["clamped"] = sorted(set(path["clamped"]) | {clamp})
        return simulate_pathway(p2, steps=steps, enzyme_scale=scale, c0=c0, record=False)

    cands = []
    base_scale = list(sick_scale)

    for lvl, name in ((0.3, "partly"), (1.0, "fully")):
        sc = list(base_scale)
        sc[j] = lvl
        cands.append({"scale": sc, "kind": "restore",
                      "move": "Restore the missing enzyme " + name,
                      "mechanism": "enzyme replacement, gene therapy, or a chaperone or vitamin that raises residual activity",
                      "target": rxn["enzyme"], "run": run_with(sc)})
    near = _neighbourhood(path, reaction_id, hops=2, cap=14) if big else set(ids)
    for i, r in enumerate(path["reactions"]):
        if i == j or r["id"] not in near:
            continue
        for f, label, mech in ((0.3, "Block", "substrate reduction or a deliberate downstream block, as nitisinone does in tyrosinemia"),
                               (2.5, "Boost", "push an alternative route, by induction or by supplying a cofactor")):
            sc = list(base_scale)
            sc[i] = f
            cands.append({"scale": sc, "kind": "modulate",
                          "move": f"{label} {r['enzyme']} ({r['name']})", "mechanism": mech,
                          "target": r["enzyme"], "reaction": r["id"], "run": run_with(sc)})
    if toxin:
        cands.append({"move": f"Drain {toxin} with a scavenger", "target": toxin,
                      "mechanism": "a drug that carries the accumulating metabolite out, as benzoate and phenylbutyrate do for ammonia",
                      "run": run_with(base_scale, extra_exit=toxin)})
    for s in rxn["products"]:
        c0 = initial_state(path)
        c0[s] = max(h.get(s, 0.5), 0.5)
        cands.append({"move": f"Supply {s} directly", "target": s,
                      "mechanism": "give the missing product, as uridine does in orotic aciduria and cholesterol in Smith-Lemli-Opitz",
                      "run": run_with(base_scale, clamp=s, c0=c0)})
    for s in path["clamped"]:
        c0 = initial_state(path)
        c0[s] = c0[s] / 3.0
        cands.append({"move": f"Restrict {s} in the diet", "target": s,
                      "mechanism": "dietary restriction, as in PKU, galactosemia and hereditary fructose intolerance",
                      "run": run_with(base_scale, c0=c0)})

    named = {}
    for r in path["reactions"]:
        for dr in r.get("drugs", []):
            named.setdefault(r["enzyme"], []).append(dr)
    rows = []
    for c in cands:
        sc, tox_after, flux_after = score(c["run"])
        real = named.get(c.get("target"), [])
        rows.append({"move": c["move"], "mechanism": c["mechanism"], "score": sc, "toxin_after": tox_after,
                     "flux_after": flux_after, "settled": c["run"]["steady"],
                     "known_drugs": [d["name"] for d in real]})
    for c, row in zip(cands, rows):
        row["_scale"] = c.get("scale")
        row["_kind"] = c.get("kind")
    rows.sort(key=lambda r: -r["score"])
    pairs = []
    if combos:
        singles = [r for r in rows if r.get("_scale") and r["_kind"] == "modulate"][:3]
        for a in range(len(singles)):
            for b in range(a + 1, len(singles)):
                sc = [min(x, y) if x < 1 or y < 1 else max(x, y)
                      for x, y in zip(singles[a]["_scale"], singles[b]["_scale"])]
                run = run_with(sc)
                s2, tox2, flux2 = score(run)
                pairs.append({"move": f"{singles[a]['move']} + {singles[b]['move']}", "score": s2,
                              "toxin_after": tox2, "flux_after": flux2, "settled": run["steady"],
                              "best_single": max(singles[a]["score"], singles[b]["score"]),
                              "known_drugs": sorted(set(singles[a]["known_drugs"]) | set(singles[b]["known_drugs"]))})
        pairs.sort(key=lambda r: -r["score"])
    off = {}
    if path.get("origin_species") and rows:
        best = next((r for r in rows if r.get("_kind") == "modulate"), None)
        if best and best.get("_scale"):
            run = run_with(best["_scale"])
            lv = _levels(run, sp)
            moved = [{"species": s2, "before": round(float(h[s2]), 3), "after": round(float(lv[s2]), 3)}
                     for s2 in sp if s2 not in path["clamped"] and abs(lv[s2] - h[s2]) > 0.05]
            off = {"move": best["move"], "pathways": affected_pathways(path, moved), "changes": moved[:10]}
    for r in rows:
        r.pop("_scale", None)
        r.pop("_kind", None)
    return {"enzyme": rxn["enzyme"], "reaction": rxn["name"], "deficiency": rxn.get("deficiency"),
            "combinations": pairs[:4], "off_target": off,
            "toxin": toxin, "toxin_healthy": round(float(h[toxin]), 3) if toxin else None,
            "toxin_disease": round(float(d[toxin]), 3) if toxin else None,
            "output": out_flux, "output_healthy": round(float(h_out), 3), "output_disease": round(float(d_out), 3),
            "residual": residual, "strategies": rows[:top], "tried": len(rows)}


def benchmark(progress=None, steps=1500):
    """Test the strategy engine against reality, with a chance baseline.

    For every enzyme with a known deficiency, knock it down, rank the moves that act on *other* enzymes, and ask
    whether a move on an enzyme a real drug targets lands in the top three. Pathways are small and many enzymes
    carry drugs, so a random ranking would often score too; the expected hits under random ranking are computed
    exactly and reported alongside, so the comparison is honest.
    """
    from math import comb
    cases = []
    jobs = [(pid, r) for pid, p in PATHWAYS.items() for r in p["reactions"] if r.get("deficiency")]
    for n, (pid, r) in enumerate(jobs):
        if progress:
            progress((n + 1) / len(jobs), f"{pid}: {r['enzyme']}")
        res = strategies(PATHWAYS[pid], r["id"], steps=steps, top=100, combos=False)
        if res is None:
            continue
        mods = [s for s in res["strategies"] if s["move"].startswith(("Block", "Boost"))]
        k, m = sum(1 for s in mods if s["known_drugs"]), len(mods)
        rank = next((i + 1 for i, s in enumerate(mods) if s["known_drugs"]), None)
        chance = (1 - comb(m - k, 3) / comb(m, 3)) if (k and m >= 3) else (1.0 if k else 0.0)
        first = next((s for s in mods if s["known_drugs"]), None)
        cases.append({"pathway": pid, "enzyme": r["enzyme"], "deficiency": r["deficiency"].split(":")[0],
                      "toxin": res["toxin"],
                      "fold": round((res["toxin_disease"] or 0) / max(res["toxin_healthy"] or 1e-6, 1e-6), 1)
                      if res["toxin"] else None,
                      "top_move": res["strategies"][0]["move"] if res["strategies"] else None,
                      "moves": m, "drug_moves": k, "rank": rank, "chance_top3": round(chance, 3),
                      "matched_move": first["move"] if first else None,
                      "drugs": sorted(set(first["known_drugs"]))[:4] if first else []})
    scored = [c for c in cases if c["drug_moves"]]
    observed = sum(1 for c in scored if c["rank"] and c["rank"] <= 3)
    expected = sum(c["chance_top3"] for c in scored)
    var = sum(c["chance_top3"] * (1 - c["chance_top3"]) for c in scored)
    z = (observed - expected) / var ** 0.5 if var > 0 else 0.0
    return {"cases": cases, "total": len(cases), "scored": len(scored), "observed_top3": observed,
            "expected_top3": round(expected, 1), "z": round(z, 2),
            "note": "Only moves that block or boost another enzyme are ranked here, since restoring the lost enzyme "
                    "is trivially right. 'Expected by chance' is what a random ranking would score on the same cases; "
                    "the method is only adding something to the extent the observed count beats it. A match means "
                    "the model favoured an enzyme that some real drug acts on, which is suggestive, not proof that "
                    "it found that drug's clinical use."}


def target_search(path, species, direction="lower", steps=4000, block=0.3, boost=3.0):
    """Drug-target search for a model: which step, blocked or boosted, moves one quantity furthest the way you want?

    Uses the average level over the whole run rather than the end point, so it works for transient responses such
    as a vaccine or a single drug dose as well as for steady states.
    """
    sp = path["species"]
    if species not in sp:
        return None
    i = sp.index(species)
    ids = [r["id"] for r in path["reactions"]]

    def avg(scale=None):
        run = simulate_pathway(path, steps=steps, enzyme_scale=scale, record=True)
        tr = np.array(run["trace"])
        return float(tr[:, i].mean()), {s: float(tr[:, k].mean()) for k, s in enumerate(sp)}

    base, base_all = avg()
    rows = []
    for j, r in enumerate(path["reactions"]):
        for f, label in ((block, "Block"), (boost, "Boost")):
            scale = [f if k == j else 1.0 for k in range(len(ids))]
            val, allv = avg(scale)
            change = (val - base) / max(abs(base), 1e-9)
            good = -change if direction == "lower" else change
            side = sorted(((s, (allv[s] - base_all[s]) / max(abs(base_all[s]), 1e-9)) for s in sp if s != species),
                          key=lambda kv: -abs(kv[1]))[:3]
            rows.append({"move": f"{label} {r['name']}", "reaction": r["id"], "factor": f, "level": round(val, 4),
                         "change": round(change, 3), "benefit": round(good, 3),
                         "side_effects": [{"species": s, "change": round(c, 3)} for s, c in side if abs(c) > 0.05],
                         "known_drugs": [d["name"] for d in r.get("drugs", [])]})
    rows.sort(key=lambda r: -r["benefit"])
    return {"species": species, "direction": direction, "baseline": round(base, 4), "moves": rows}
