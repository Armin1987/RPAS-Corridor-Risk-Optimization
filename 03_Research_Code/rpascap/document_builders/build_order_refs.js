// Evaluate the manuscript section modules in document order so that citation numbering
// (order of first citation) is identical in the manuscript and the supplementary table.
module.exports = (D) => {
  const core = require("./ms_core")(D);
  core.intro(); core.lit(); core.problem(); core.model(); require("./ms_data")(D); core.method(); require("./ms_results")(D);
};
