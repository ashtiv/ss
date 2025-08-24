import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import warnings, logging, os
from tabulate import tabulate

# -----------------------
# Suppress warnings/logs
# -----------------------
warnings.filterwarnings('ignore')
logging.getLogger('lightgbm').setLevel(logging.CRITICAL)
os.environ['LIGHTGBM_VERBOSITY'] = 'FATAL'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

days_array = [700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
lookbacks = [30, 60, 90]

def prepare_dataset(data, lookback=30):
    X, y = [], []
    for i in range(lookback, len(data)-1):
        X.append(data[i-lookback:i])
        y.append(1 if data[i+1] > data[i] else 0)
    X = np.array(X).reshape(len(X), lookback)
    y = np.array(y)
    return X, y

def evaluate_best_model(symbol):
    try:
        df = yf.download(symbol, period='10y', interval='1d')
        if df.empty:
            print(f"⚠️ No data for {symbol}, skipping.")
            return None
        data_full = df["Close"].values
        dates_full = df.index
    except Exception as e:
        print(f"❌ Failed download for {symbol}: {e}")
        return None

    best_result = None
    for days in days_array:
        data = data_full[-days:]
        date_subset = dates_full[-days:]
        for lb in lookbacks:
            if lb >= len(data)-1:
                continue
            X, y = prepare_dataset(data, lb)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            model = LGBMClassifier(verbose=-1)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            if (best_result is None) or (acc > best_result['accuracy']):
                best_result = {
                    'symbol': symbol,
                    'days': days,
                    'lookback': lb,
                    'accuracy': acc,
                    'model': model,
                    'X_last': X[-1],
                    'last_day_date': date_subset[-1],
                    'actual_last_dir': "UP" if data[-1] > data[-2] else "DOWN"
                }

    if best_result is None:
        print(f"⚠️ No valid model for {symbol}")
        return None

    # Next day prediction
    last_lb_days = data_full[-best_result['lookback']:]
    next_pred = best_result['model'].predict(last_lb_days.reshape(1, -1))[0]
    best_result['next_day_pred'] = "UP" if next_pred == 1 else "DOWN"

    # Backtest prediction
    backtest_pred = best_result['model'].predict(best_result['X_last'].reshape(1,-1))[0]
    best_result['backtest_pred'] = "UP" if backtest_pred == 1 else "DOWN"
    best_result['backtest_date'] = best_result['last_day_date'].strftime('%Y-%m-%d')

    return best_result

if __name__ == "__main__":
    # Only the symbols that previously failed
    symbols = ['RELIANCE.NS', 'HDFCBANK.NS', 'BHARTIARTL.NS', 'ICICIBANK.NS', 'SBIN.NS', 'INFY.NS', 'HINDUNILVR.NS', 'LICI.NS', 'BAJFINANCE.NS', 'ITC.NS', 'KOTAKBANK.NS', 'SUNPHARMA.NS', 'AXISBANK.NS', 'NTPC.NS', 'BAJAJFINSV.NS', 'ETERNAL.NS', 'ONGC.NS', 'ADANIPORTS.NS', 'BEL.NS', 'ADANIENT.NS', 'POWERGRID.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'ADANIPOWER.NS', 'TATASTEEL.NS', 'IOC.NS', 'HYUNDAI.NS', 'GRASIM.NS', 'DLF.NS', 'SBILIFE.NS', 'VBL.NS', 'VEDL.NS', 'HDFCLIFE.NS', 'HINDALCO.NS', 'ADANIGREEN.NS', 'AMBUJACEM.NS', 'BPCL.NS', 'PFC.NS', 'CIPLA.NS', 'GODREJCP.NS', 'CHOLAFIN.NS', 'LODHA.NS', 'BANKBARODA.NS', 'MAXHEALTH.NS', 'GAIL.NS', 'SHRIRAMFIN.NS', 'INDHOTEL.NS', 'MAZDOCK.NS', 'MUTHOOTFIN.NS', 'MANKIND.NS', 'TATACONSUM.NS', 'SWIGGY.NS', 'CGPOWER.NS', 'UNIONBANK.NS', 'JINDALSTEL.NS', 'MOTHERSON.NS', 'CANBK.NS', 'RECLTD.NS', 'HAVELLS.NS', 'UNITDSPR.NS', 'ADANIENSOL.NS', 'ICICIGI.NS', 'BSE.NS', 'MARICO.NS', 'GMRAIRPORT.NS', 'ICICIPRULI.NS', 'JSWENERGY.NS', 'INDIANB.NS', 'LUPIN.NS', 'NAUKRI.NS', 'HINDPETRO.NS', 'SUZLON.NS', 'ASHOKLEY.NS', 'ABCAPITAL.NS', 'UNOMINDA.NS', 'COROMANDEL.NS', 'GVT&D.NS', 'LLOYDSME.NS', 'FORTIS.NS', 'PRESTIGE.NS', 'VMM.NS', 'OIL.NS', 'PATANJALI.NS', 'NYKAA.NS', 'JSWINFRA.NS', 'JSL.NS', 'GODREJPROP.NS', 'NMDC.NS', 'AUROPHARMA.NS', 'COFORGE.NS', 'AIRTELPP.NS', 'MFSL.NS', 'PHOENIXLTD.NS', 'MOTILALOFS.NS', 'BDL.NS', 'LTF.NS', 'NAM-INDIA.NS', 'GLENMARK.NS', 'MPHASIS.NS', 'UPL.NS', 'KALYANKJIL.NS', 'BANKINDIA.NS', 'ITCHOTELS.NS', 'HEXT.NS', 'FEDERALBNK.NS', 'BIOCON.NS', 'APLAPOLLO.NS', 'TATACOMM.NS', 'GODREJIND.NS', '360ONE.NS', 'HUDCO.NS', 'MAHABANK.NS', 'ENDURANCE.NS', 'RADICO.NS', 'ASTRAL.NS', 'MEDANTA.NS', 'NH.NS', 'POONAWALLA.NS', 'IPCALAB.NS', 'DELHIVERY.NS', 'CHOLAHLDNG.NS', 'ACC.NS', 'GODIGIT.NS', 'COHANCE.NS', 'KPITTECH.NS', 'NLCINDIA.NS', 'METROBRAND.NS', 'ASTERDM.NS', 'LICHSGFIN.NS', 'SUMICHEM.NS', 'APOLLOTYRE.NS', 'KIMS.NS', 'GRSE.NS', 'IGL.NS', 'EMCURE.NS', 'NBCC.NS', 'SONACOMS.NS', 'MSUMI.NS', 'AFFLE.NS', 'JBCHEPHARM.NS', 'SYNGENE.NS', 'EMAMILTD.NS', 'IRB.NS', 'SHYAMMETL.NS', 'AEGISLOG.NS', 'FSL.NS', 'PPLPHARMA.NS', 'ABSLAMC.NS', 'EIHOTEL.NS', 'INOXWIND.NS', 'ERIS.NS', 'BRIGADE.NS', 'WELCORP.NS', 'HSCL.NS', 'ANGELONE.NS', 'AADHARHFC.NS', 'CREDITACC.NS', 'CHALET.NS', 'KPIL.NS', 'CESC.NS', 'KEC.NS', 'JYOTICNC.NS', 'SAGILITY.NS', 'ONESOURCE.NS', 'SUNDRMFAST.NS', 'EIDPARRY.NS', 'PNBHOUSING.NS', 'KARURVYSYA.NS', 'DEVYANI.NS', 'SCHNEIDER.NS', 'CROMPTON.NS', 'ABREL.NS', 'FIRSTCRY.NS', 'KFINTECH.NS', 'SAILIFE.NS', 'DEEPAKFERT.NS', 'BIKAJI.NS', 'IIFL.NS', 'ANANTRAJ.NS', 'CGCL.NS', 'TECHNOE.NS', 'PARADEEP.NS', 'UTIAMC.NS', 'ACMESOLAR.NS', 'VENTIVE.NS', 'VINATIORGA.NS', 'FIVESTAR.NS', 'APTUS.NS', 'JUBLPHARMA.NS', 'GABRIEL.NS', 'TRITURBINE.NS', 'RATNAMANI.NS', 'WHIRLPOOL.NS', 'CAPLIPOINT.NS', 'CENTURYPLY.NS', 'VGUARD.NS', 'RAINBOW.NS', 'PGEL.NS', 'AFCONS.NS', 'ELGIEQUIP.NS', 'SOBHA.NS', 'NIVABUPA.NS', 'SIGNATURE.NS', 'CUB.NS', 'TBOTEK.NS', 'ATHERENERG.NS', 'IGIL.NS', 'CIEINDIA.NS', 'LTFOODS.NS', 'DOMS.NS', 'TARIL.NS', 'SPLPETRO.NS', 'ABDL.NS', 'TRIDENT.NS', 'DATAPATTNS.NS', 'AGARWALEYE.NS', 'PCBL.NS', 'GODREJAGRO.NS', 'KSB.NS', 'THELEELA.NS', 'AARTIIND.NS', 'GESHIP.NS', 'KIRLOSENG.NS', 'RRKABEL.NS', 'NCC.NS', 'HOMEFIRST.NS', 'BLUEJET.NS', 'INTELLECT.NS', 'LEMONTREE.NS', 'SYRMA.NS', 'NETWEB.NS', 'FINPIPE.NS', 'MGL.NS', 'AAVAS.NS', 'BELRISE.NS', 'FINCABLES.NS', 'JINDALSAW.NS', 'GRAVITA.NS', 'ZYDUSWELL.NS', 'JYOTHYLAB.NS', 'ELECON.NS', 'TEGA.NS', 'SBFC.NS', 'RITES.NS', 'GRINFRA.NS', 'CLEAN.NS', 'MINDACORP.NS', 'CELLO.NS', 'ALIVUS.NS', 'BALRAMCHIN.NS', 'CCL.NS', 'ACUTAAS.NS', 'TITAGARH.NS', 'SHRIPISTON.NS', 'BLACKBUCK.NS', 'USHAMART.NS', 'JUBLINGREA.NS', 'STARCEMENT.NS', 'WELSPUNLIV.NS', 'J&KBANK.NS', 'METROPOLIS.NS', 'SUDARSCHEM.NS', 'CARTRADE.NS', 'GRANULES.NS', 'PVRINOX.NS', 'IXIGO.NS', 'GENUSPOWER.NS', 'TIMETECHNO.NS', 'VESUVIUS.NS', 'ENGINERSIN.NS', 'EUREKAFORB.NS', 'SAPPHIRE.NS', 'GRAPHITE.NS', 'TRANSRAILL.NS', 'INOXINDIA.NS', 'SONATSOFTW.NS', 'AZAD.NS', 'CANFINHOME.NS', 'AETHER.NS', 'NESCO.NS', 'ASKAUTOLTD.NS', 'RHIM.NS', 'BIRLACORPN.NS', 'WABAG.NS', 'SAFARI.NS', 'MEDPLUS.NS', 'INDIASHLTR.NS', 'ASTRAMICRO.NS', 'TI.NS', 'HCG.NS', 'MAPMYINDIA.NS', 'HEG.NS', 'JAIBALAJI.NS', 'JLHL.NS', 'HAPPSTMNDS.NS', 'MAHSEAMLES.NS', 'HAPPYFORGE.NS', 'JKTYRE.NS', 'TCI.NS', 'AVANTIFEED.NS', 'SHARDACROP.NS', 'LATENTVIEW.NS', 'GMRP&UI.NS', 'UJJIVANSFB.NS', 'BECTORFOOD.NS', 'KIRLPNU.NS', 'VARROC.NS', 'TANLA.NS', 'RUSTOMJEE.NS', 'MARKSANS.NS', 'CAMPUS.NS', 'DODLA.NS', 'BBOX.NS', 'THOMASCOOK.NS', 'GARFIBRES.NS', 'SANSERA.NS', 'STAR.NS', 'EPIGRAL.NS', 'PNCINFRA.NS', 'PNGJL.NS', 'GALAXYSURF.NS', 'LUMAXTECH.NS', 'AJAXENGG.NS', 'MASTEK.NS', 'ARVIND.NS', 'EPL.NS', 'AKUMS.NS', 'AURIONPRO.NS', 'TIPSMUSIC.NS', 'TRIVENI.NS', 'MAXESTATES.NS', 'MAHLIFE.NS', 'MIDHANI.NS', 'MHRIL.NS', 'CMSINFO.NS', 'DHANUKA.NS', 'JUSTDIAL.NS', 'RAYMONDLSL.NS', 'CSBBANK.NS', 'ARVINDFASN.NS', 'CHEMPLASTS.NS', 'PURVA.NS', 'ETHOSLTD.NS', 'THYROCARE.NS', 'BAJAJELEC.NS', 'MOIL.NS', 'JUNIPER.NS', 'SYMPHONY.NS', 'THANGAMAYL.NS', 'KTKBANK.NS', 'HGINFRA.NS', 'SWSOLAR.NS', 'AVL.NS', 'TIIL.NS', 'EQUITASBNK.NS', 'AHLUCONT.NS', 'SUPRAJIT.NS', 'PGIL.NS', 'HNDFDS.NS', 'EMUDHRA.NS', 'SENCO.NS', 'GULFOILLUB.NS', 'SHARDAMOTR.NS', 'KSCL.NS', 'IFBIND.NS', 'VMART.NS', 'GREENLAM.NS', 'JKPAPER.NS', 'RATEGAIN.NS', 'TVSSCS.NS', 'BANSALWIRE.NS', 'TEXRAIL.NS', 'SUNTECK.NS', 'MASFIN.NS', 'SUBROS.NS', 'PTC.NS', 'SUPRIYA.NS', 'GOKEX.NS', 'PARAS.NS', 'UNIMECH.NS', 'GMMPFAUDLR.NS', 'PRICOLLTD.NS', 'ZAGGLE.NS', 'ENTERO.NS', 'FEDFINA.NS', 'SIS.NS', 'INNOVACAP.NS', 'EMIL.NS', 'HCC.NS', 'BALAMINES.NS', 'VRLLOG.NS', 'JKIL.NS', 'GAEL.NS', 'SPARC.NS', 'DBCORP.NS', 'TARC.NS', 'MTARTECH.NS', 'RBA.NS', 'SAMHI.NS', 'PDSL.NS', 'ANUP.NS', 'ORIENTELEC.NS', 'NIITMTS.NS', 'GOPAL.NS', 'INFIBEAM.NS', 'SJS.NS', 'SANATHAN.NS', 'INDRAMEDCO.NS', 'RAYMOND.NS', 'LGBBROSLTD.NS', 'BOROLTD.NS', 'AWFIS.NS', 'JAMNAAUTO.NS', 'WONDERLA.NS', 'QUESS.NS', 'MEDIASSIST.NS', 'ORCHPHARMA.NS', 'GREENPLY.NS', 'DCBBANK.NS', 'PRINCEPIPE.NS', 'NEOGEN.NS', 'GOCOLORS.NS', 'INDOSTAR.NS', 'KOLTEPATIL.NS', 'EPACK.NS', 'GLOBUSSPR.NS', 'PITTIENG.NS', 'CYIENTDLM.NS', 'DEEPINDS.NS', 'ROSSARI.NS', 'VISHNU.NS', 'PENIND.NS', 'GANECOS.NS', 'GATEWAY.NS', 'ADVENZYMES.NS', 'JASH.NS', 'INTERARCH.NS', 'DDEVPLSTIK.NS', 'SENORES.NS', 'SSWL.NS', 'ALLCARGO.NS', 'PATELENG.NS', 'FLAIR.NS', 'PARKHOTELS.NS', 'IMAGICAA.NS', 'EVEREADY.NS', 'HPL.NS', 'SAGCEM.NS', 'AJANTPHARM.NS', 'SHK.NS', 'BAJAJCON.NS', 'TEAMLEASE.NS', 'SBCL.NS', 'KDDL.NS', 'JTLIND.NS', 'DCXINDIA.NS', 'FCL.NS', 'GIPCL.NS', 'NRBBEARING.NS', 'ARVSMART.NS', 'KALAMANDIR.NS', 'IOLCP.NS', 'PARAGMILK.NS', 'LAOPALA.NS', 'STYLAMIND.NS', 'NUCLEUS.NS', 'PAISALO.NS', 'AUTOAXLES.NS', 'VENUSPIPES.NS', 'PSPPROJECT.NS', 'SANDHAR.NS', 'MOLDTKPAC.NS', 'JTEKTINDIA.NS', 'MANINDS.NS', 'KRSNAA.NS', 'CARYSIL.NS', 'CAPACITE.NS', 'HINDWAREAP.NS', 'INSECTICID.NS', 'YATRA.NS', 'LANDMARK.NS', 'MAYURUNIQ.NS', 'STYLEBAAZA.NS', 'AEROFLEX.NS', 'STOVEKRAFT.NS', 'FINOPB.NS', 'SHALBY.NS', 'DIVGIITTS.NS', 'APCOTEXIND.NS', 'SOMANYCERA.NS', 'INDIANHUME.NS', 'UGROCAP.NS', 'NITINSPIN.NS', 'LAXMIDENTL.NS', 'IFGLEXPOR.NS', 'SPAL.NS', 'GENESYS.NS', 'APOLLOPIPE.NS', 'MOBIKWIK.NS', 'SPANDANA.NS', 'HITECH.NS', 'ECOSMOBLTY.NS', 'TARSONS.NS', 'UDS.NS', 'AWHCL.NS', 'RAMCOSYS.NS', 'ADOR.NS', 'MMFL.NS', 'SURAJEST.NS', 'UNIECOM.NS', 'SURYODAY.NS', 'ALICON.NS', 'CAPITALSFB.NS', 'GNA.NS', 'SIRCA.NS', 'AURUM.NS', 'GRPLTD.NS', 'LINCOLN.NS', 'BIL.NS', 'ELIN.NS', 'TVTODAY.NS', 'BODALCHEM.NS', 'DREAMFOLKS.NS', 'RATNAVEER.NS', 'DWARKESH.NS', 'ENIL.NS', 'KILITCH.NS', 'SCHAND.NS', 'INFIBPP.NS', 'VENUSREM.NS', 'SSFLPP.NS']   
    results = []

    for sym in symbols:
        print(f"\nProcessing {sym} ...")
        best = evaluate_best_model(sym)
        if best:
            results.append(best)

    if results:
        # Sort by accuracy descending
        results.sort(key=lambda x: x['accuracy'], reverse=True)

        # Prepare table
        table = []
        for r in results:
            table.append([
                r['symbol'],
                r['days'],
                r['lookback'],
                f"{r['accuracy']:.4f}",
                f"{r['backtest_date']} ({r['backtest_pred']} / actual: {r['actual_last_dir']})",
                r['next_day_pred']
            ])

        headers = ["Symbol", "Days", "Lookback", "Accuracy", "Backtest (Pred / Actual)", "Next day pred"]
        print("\n" + tabulate(table, headers=headers, tablefmt="pretty"))
    else:
        print("⚠️ No symbols could be processed.")
