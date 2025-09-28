import itertools
import json
import time
import numpy as np

from create_model_summary import run_r_iterations
from process_data.main import load_dataframe, FeatureLabelReducer

if __name__ == "__main__":
    base_file_path = "../data/CGMacros/pickle/"
    df_dict = dict()
    for pkl in ["cgm", "dynamic_user", "log", "static_user"]:
        df_dict[pkl] = load_dataframe(base_file_path + pkl + ".pkl")

    # ----------------------------------- #
    feature_groups = {
        "static_user": ["Sex", "Body weight", "Height", "Self-identity"],
        "static_user2": ["UserID"],
        "log_macros": ["Energy", "Carbohydrate", "Protein", "Fat"],
        "log_food": ["Food Types"],
        "temporal_cgm": ["cgm_p30", "cgm_p60", "cgm_p120"],
        "temporal_food": ["meal_hour", "time_since_last_meal"]
    }

    feature_key_mapping = {
        "static_user": "Su",
        "static_user2": "Ui",
        "log_macros": "Fm",
        "temporal_cgm": "Tg",
        "temporal_food": "Tf",
        "log_food": "Ft"
    }

    # Get all combinations of inclusion (True/False)
    combinations = list(itertools.product([False, True], repeat=len(feature_groups)))
    results = []

    for comb in combinations:
        include_groups = {k: v for k, v, inc in zip(feature_groups.keys(), feature_groups.values(), comb) if inc}
        if len(include_groups) == 0:
            continue
        nickname = "".join([feature_key_mapping[k] for k, inc in zip(feature_groups.keys(), comb) if inc])

        t1 = time.time()
        # Get X, Y for this combinations
        reducer = FeatureLabelReducer(df_dict, include_groups)
        feature_names, x_values, y_values = reducer.get_x_y_data()

        # Run your experiment
        rs = run_r_iterations(x_values, y_values, r_iterations=100, print_updates=True)
        t2 = time.time()
        # Save results with nickname
        results.append({"nickname": nickname, "result": rs, "time": t2 - t1})
        print(nickname, np.mean(rs), t2 - t1)

    # print(results)

    print(results)
    with open("data.json", "w") as f:
        json.dump(results, f, indent=4)  # indent=4 makes it pretty-printed

    # results = [{'nickname': 'Tf', 'result': [np.float64(-0.03991407150685172), np.float64(0.08325178503018843), np.float64(0.012994361016031514), np.float64(-0.1356792258589937), np.float64(-0.08805812584001256)]}, {'nickname': 'Tg', 'result': [np.float64(0.05415001630093367), np.float64(0.27826084942775264), np.float64(0.3059409004837648), np.float64(0.2186427668702357), np.float64(0.30008838684658756)]}, {'nickname': 'TgTf', 'result': [np.float64(0.05788818095998949), np.float64(0.06615968897462618), np.float64(0.21469041602232022), np.float64(0.2510936695859786), np.float64(0.610050389354674)]}, {'nickname': 'Ft', 'result': [np.float64(0.21066554473072432), np.float64(0.19221079750569012), np.float64(0.09756467617416587), np.float64(0.15295387519718687), np.float64(0.022163200383750286)]}, {'nickname': 'FtTf', 'result': [np.float64(0.19137877488747934), np.float64(-0.022827100109773117), np.float64(0.21272539698260012), np.float64(-0.0392159821398891), np.float64(0.0932546116401288)]}, {'nickname': 'FtTg', 'result': [np.float64(0.3090827779044201), np.float64(0.30714119847134247), np.float64(0.0670808139465903), np.float64(0.2073901744967475), np.float64(0.12190741916406204)]}, {'nickname': 'FtTgTf', 'result': [np.float64(0.12538767201120438), np.float64(0.41226649244776276), np.float64(0.3305408302514819), np.float64(0.4681336557003059), np.float64(0.28252264042292086)]}, {'nickname': 'Fm', 'result': [np.float64(0.32085766694474854), np.float64(0.10616184425226857), np.float64(0.1636373003915522), np.float64(0.5388711521218978), np.float64(0.11777418038164227)]}, {'nickname': 'FmTf', 'result': [np.float64(0.1158834296995458), np.float64(0.28031845652747106), np.float64(0.34415337828702514), np.float64(0.27067543399289884), np.float64(0.36762047232632034)]}, {'nickname': 'FmTg', 'result': [np.float64(0.4505466403424017), np.float64(0.3237163544126005), np.float64(0.5641874742746392), np.float64(0.36556726819787827), np.float64(0.40504705619376274)]}, {'nickname': 'FmTgTf', 'result': [np.float64(0.438291753679757), np.float64(0.26646023068604585), np.float64(0.2238174780659189), np.float64(0.5346597574208353), np.float64(0.39375614005584186)]}, {'nickname': 'FmFt', 'result': [np.float64(0.2503259097340074), np.float64(0.45123800567146233), np.float64(0.2056034170152924), np.float64(0.4302466592429896), np.float64(0.24940885765400683)]}, {'nickname': 'FmFtTf', 'result': [np.float64(0.3682178320021303), np.float64(0.23697657027751193), np.float64(0.3158366574712998), np.float64(0.31466450619405173), np.float64(0.3454808580061485)]}, {'nickname': 'FmFtTg', 'result': [np.float64(0.37980096860520124), np.float64(0.4909934424890225), np.float64(0.4277506048571484), np.float64(0.5115868326076488), np.float64(0.45527400920218597)]}, {'nickname': 'FmFtTgTf', 'result': [np.float64(0.4761052635544236), np.float64(0.5770071834599378), np.float64(0.5305215619479352), np.float64(0.5262912952508336), np.float64(0.3531581518245152)]}, {'nickname': 'Su', 'result': [np.float64(0.6096179749135853), np.float64(0.5259438177855655), np.float64(0.38897560704547063), np.float64(0.5071217783100788), np.float64(0.44926185996827095)]}, {'nickname': 'SuTf', 'result': [np.float64(0.38053249431010056), np.float64(0.5944296248676301), np.float64(0.5690580340079159), np.float64(0.3918054596747513), np.float64(0.16652911397869066)]}, {'nickname': 'SuTg', 'result': [np.float64(0.4196022203465399), np.float64(0.4602474302239341), np.float64(0.5586864928856384), np.float64(0.6303416407576602), np.float64(0.42600300382959005)]}, {'nickname': 'SuTgTf', 'result': [np.float64(0.49355681072081364), np.float64(0.6278715964279539), np.float64(0.48424761431433705), np.float64(0.48797397723443775), np.float64(0.6565129768103631)]}, {'nickname': 'SuFt', 'result': [np.float64(0.4282697912855884), np.float64(0.30887543981353816), np.float64(0.5160298150857655), np.float64(0.5477489713072609), np.float64(0.3734595181647137)]}, {'nickname': 'SuFtTf', 'result': [np.float64(0.5878322095809405), np.float64(0.5243171627162558), np.float64(0.4000352672369073), np.float64(0.45019712958215025), np.float64(0.573776589012647)]}, {'nickname': 'SuFtTg', 'result': [np.float64(0.42907772689900076), np.float64(0.4582782059420422), np.float64(0.4401556580323002), np.float64(0.5507635493966921), np.float64(0.5366943500052218)]}, {'nickname': 'SuFtTgTf', 'result': [np.float64(0.5927934080238287), np.float64(0.6882132168302996), np.float64(0.7007816993249217), np.float64(0.5134461723588905), np.float64(0.48857817758806027)]}, {'nickname': 'SuFm', 'result': [np.float64(0.6552164664340833), np.float64(0.5419642937986433), np.float64(0.5395297277774403), np.float64(0.5305846257610156), np.float64(0.6233152806881594)]}, {'nickname': 'SuFmTf', 'result': [np.float64(0.6236993215234945), np.float64(0.6081958488276032), np.float64(0.6982000884698312), np.float64(0.6991278548532026), np.float64(0.7881597057589023)]}, {'nickname': 'SuFmTg', 'result': [np.float64(0.7303774127273347), np.float64(0.6352024388075439), np.float64(0.7740507104810953), np.float64(0.5352079741628666), np.float64(0.6359576479247773)]}, {'nickname': 'SuFmTgTf', 'result': [np.float64(0.6973566938238239), np.float64(0.5725305787948455), np.float64(0.7199670109820782), np.float64(0.7163326518925252), np.float64(0.7032968792254269)]}, {'nickname': 'SuFmFt', 'result': [np.float64(0.3651487755788459), np.float64(0.5639869113769974), np.float64(0.3808535483797729), np.float64(0.6703014010883346), np.float64(0.7490540533132678)]}, {'nickname': 'SuFmFtTf', 'result': [np.float64(0.5966393440548325), np.float64(0.7433959750450226), np.float64(0.5509812866970198), np.float64(0.4350253895596028), np.float64(0.6111061073528306)]}, {'nickname': 'SuFmFtTg', 'result': [np.float64(0.7484914591762603), np.float64(0.5892151650489712), np.float64(0.31331228634477365), np.float64(0.614952655151367), np.float64(0.557886522642825)]}, {'nickname': 'SuFmFtTgTf', 'result': [np.float64(0.5084110265833943), np.float64(0.5962101955751297), np.float64(0.6280303825748824), np.float64(0.811960371546551), np.float64(0.6652793125402912)]}]
    #
    # li = []
    # for d in results:
    #     li.append({"nickname": d["nickname"], "R": float(np.mean(d["result"]))})
    # print(li)
