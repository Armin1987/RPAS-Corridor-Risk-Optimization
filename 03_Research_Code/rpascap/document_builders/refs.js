// Verified reference registry.  Each entry was checked in this project on a publisher,
// DOI-registry, repository, RePEc, arXiv, Zenodo or official page (see Supplementary S9).
// cite('key') numbers references in order of first citation (Springer Nature numbered style).
const R = {
  pneuma2020: { text: "Barmpounakis E, Geroliminis N (2020) On the new era of urban traffic monitoring with massive drone data: the pNEUMA large-scale field experiment. Transp Res Part C 111:50–71", doi: "10.1016/j.trc.2019.11.023" },
  pneumaData: { text: "Barmpounakis E, Geroliminis N (2024) pNEUMA dataset (version 1). Zenodo", doi: "10.5281/zenodo.10491409" },
  kimbae: { text: "Kim Y, Bae J (2022) Risk-based UAV corridor capacity analysis above a populated area. Drones 6(9):221", doi: "10.3390/drones6090221" },
  hou: { text: "Hou W, Fang T, Pei Z, He Q-C (2021) Integrated design of unmanned aerial mobility network: a data-driven risk-averse approach. Int J Prod Econ 236:108131", doi: "10.1016/j.ijpe.2021.108131" },
  pang2024: { text: "Pang B, Hu X, Dai W, Low KH (2024) Stochastic route optimization under dynamic ground risk uncertainties for safe drone delivery operations. Transp Res Part E 192:103717", doi: "10.1016/j.tre.2024.103717" },
  yao: { text: "Yao Y, Ding L, Zhao Y (2026) Stochastic corridor–time network capacity planning for low-altitude airspace systems. arXiv preprint arXiv:2608.11477", url: "https://arxiv.org/abs/2608.11477" },
  zhu: { text: "Zhu Y, Sun X, Hou T (2026) Adaptive airspace allocation model for urban drone logistics using multi-objective optimization under uncertainty. Sci Rep 16:2562", doi: "10.1038/s41598-025-32450-8" },
  paul: { text: "Paul A, Levin MW, Waller ST, Rey D. Data-driven optimization for drone delivery service planning with online demand. arXiv preprint arXiv:2404.02442", url: "https://arxiv.org/abs/2404.02442" },
  bk: { text: "Bertsimas D, Kallus N (2020) From predictive to prescriptive analytics. Manag Sci 66(3):1025–1044", doi: "10.1287/mnsc.2018.3253" },
  sadana: { text: "Sadana U, Chenreddy A, Delage E, Forel A, Frejinger E, Vidal T (2025) A survey of contextual optimization methods for decision-making under uncertainty. Eur J Oper Res 320(2):271–289", url: "https://econpapers.repec.org/RePEc:eee:ejores:v:320:y:2025:i:2:p:271-289" },
  reich: { text: "Reich PG (1966) Analysis of long-range air traffic systems: separation standards—I. J Navig 19(1):88–98", doi: "10.1017/s037346330004056x" },
  lacour: { text: "la Cour-Harbo A (2019) Quantifying risk of ground impact fatalities for small unmanned aircraft. J Intell Robot Syst 93:367–384", doi: "10.1007/s10846-018-0853-1" },
  primatesta: { text: "Primatesta S, Rizzo A, la Cour-Harbo A (2020) Ground risk map for unmanned aircraft in urban environments. J Intell Robot Syst 97:489–509", doi: "10.1007/s10846-019-01015-z" },
  washington: { text: "Washington A, Clothier RA, Silva J (2017) A review of unmanned aircraft system ground risk models. Prog Aerosp Sci 95:24–44", doi: "10.1016/j.paerosci.2017.10.001" },
  dalamagkidis: { text: "Dalamagkidis K, Valavanis KP, Piegl LA (2008) Evaluating the risk of unmanned aircraft ground impacts. In: 16th Mediterranean Conference on Control and Automation, Ajaccio, pp 709–716", url: "https://archive.air.cit.tum.de/Main/Publications/Dalamagkidis2008c.pdf" },
  geister: { text: "Geister R, Limmer L, Rippl M, Dautermann T (2018) Total system error performance of drones for an unmanned PBN concept. In: 2018 Integrated Communications, Navigation, Surveillance Conference (ICNS)", url: "https://elib.dlr.de/119407/" },
  pang2022: { text: "Pang B, Zhang M, Deng C, Low KH (2022) Investigation of flight technical error for UAV separation requirement based on flight trajectory data. AIAA AVIATION 2022 Forum, AIAA 2022-3763", doi: "10.2514/6.2022-3763" },
  jung: { text: "Jung J, D'Souza SN, Johnson MA, Ishihara AK, Modi HC, Nikaido B, Hasseeb H (2016) Applying required navigation performance concept for traffic management of small unmanned aircraft systems. In: 30th Congress of the International Council of the Aeronautical Sciences (ICAS)", url: "https://ntrs.nasa.gov/citations/20160011496" },
  icao: { text: "ICAO NACC Regional Office (2018) Collision risk assessment working paper GTE/18-WP/05 (citing ICAO Doc 9574)", url: "https://www2023.icao.int/NACC/Documents/Meetings/2018/GTE18/GTE18-WP05.pdf" },
  era5: { text: "Hersbach H et al (2020) The ERA5 global reanalysis. Q J R Meteorol Soc 146:1999–2049", doi: "10.1002/qj.3803" },
  saa: { text: "Kleywegt AJ, Shapiro A, Homem-de-Mello T (2002) The sample average approximation method for stochastic discrete optimization. SIAM J Optim 12(2):479–502", doi: "10.1137/s1052623499363220" },
  highs: { text: "Huangfu Q, Hall JAJ (2018) Parallelizing the dual revised simplex method. Math Program Comput 10(1):119–142", doi: "10.1007/s12532-017-0130-5" },
  scipy: { text: "Virtanen P et al (2020) SciPy 1.0: fundamental algorithms for scientific computing in Python. Nat Methods 17:261–272", doi: "10.1038/s41592-019-0686-2" },
  dji: { text: "DJI (2026) Matrice 300 RTK specifications. https://www.dji.com/au/support/product/matrice-300 (accessed 7 November 2026)", url: "https://www.dji.com/au/support/product/matrice-300" },
  openmeteo: { text: "Open-Meteo (2026) Previous Runs API, Historical Forecast API and Historical Weather API documentation. https://open-meteo.com/en/docs (accessed 7 November 2026)", url: "https://open-meteo.com/en/docs" },
  osm: { text: "OpenStreetMap contributors (2026) OpenStreetMap. Data available under the Open Database License. https://www.openstreetmap.org", url: "https://www.openstreetmap.org" },
};

const order = [];
function cite(...keys) {
  const nums = keys.map(k => {
    if (!R[k]) throw new Error("unknown reference " + k);
    let i = order.indexOf(k);
    if (i < 0) { order.push(k); i = order.length - 1; }
    return i + 1;
  });
  nums.sort((a, b) => a - b);
  // compress ranges
  const parts = []; let s = nums[0], p = nums[0];
  for (let j = 1; j <= nums.length; j++) {
    if (j < nums.length && nums[j] === p + 1) { p = nums[j]; continue; }
    parts.push(s === p ? `${s}` : (p === s + 1 ? `${s}, ${p}` : `${s}–${p}`));
    if (j < nums.length) { s = p = nums[j]; }
  }
  return "[" + parts.join(", ") + "]";
}
function list() {
  return order.map((k, i) => ({ n: i + 1, key: k, text: R[k].text, link: R[k].doi ? "https://doi.org/" + R[k].doi : R[k].url }));
}
module.exports = { cite, list, R };
