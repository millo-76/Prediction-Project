import { useEffect, useState } from "react";
import { getTeams, predictMatch } from "./api/predictorApi";
import { motion, AnimatePresence } from "framer-motion";

function App() {
  const [teams, setTeams] = useState<string[]>([]);
  const [blueTeam, setBlueTeam] = useState("");
  const [redTeam, setRedTeam] = useState("");
  const [result, setResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    async function loadTeams() {
      const teamList = await getTeams();
      setTeams(teamList);
    }

    loadTeams();
  }, []);

  async function handlePredict() {
    if (!blueTeam || !redTeam) return;

    setIsLoading(true);
    setResult(null);

    try {
      const data = await predictMatch(blueTeam, redTeam);
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Prediction failed.");
    } finally {
      setIsLoading(false);
    }
  }

  const bluePct = result ? result.blue_win_probability * 100 : 0;
  const redPct = result ? result.red_win_probability * 100 : 0;

  return (
    <main className="min-h-screen bg-black px-6 py-8 text-white">
      <section className="mx-auto flex min-h-[calc(100vh-4rem)] max-w-6xl flex-col justify-center">
        <div className="mb-10 text-center uppercase tracking-[0.35em] text-slate-500">
          Machine Learning Match Predictor
        </div>

        <h1 className="mb-12 text-center text-5xl font-bold uppercase tracking-[0.18em] text-white md:text-7xl">
          League of Legends Predictor
        </h1>

        <div className="relative z-0 grid items-center gap-8 md:grid-cols-[1fr_auto_1fr]">       
          <div className="relative">
            {blueTeam && (
              <div className="absolute inset-[-35px] -z-10 bg-[#00d1ff]/35 blur-3xl" />
            )}

            <motion.div
              initial={{ opacity: 0, x: -60 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.45}}
              whileHover={{ scale: 1.02 }}
              className={`
                blue-cyber-panel
                border-2
                border-[#00d1ff]
                bg-zinc-950
                p-6
                transition-all
                duration-500
                ${
                  blueTeam
                    ? "shadow-[0_0_60px_rgba(0,209,255,0.28)]"
                    : "shadow-[0_0_18px_rgba(0,209,255,0.10)]"
                }
              `}
              >
                <p className="mb-4 text-sm font-bold uppercase tracking-[0.3em] text-[#00d1ff]">
                  Blue Team
                </p>

              <select
                className="w-full border border-[#00d1ff]/70 bg-black p-4 text-xl font-bold uppercase tracking-widest text-white outline-none shadow-[inset_0_0_16px_rgba(0,209,255,0.25)]"
                value={blueTeam}
                onChange={(e) => setBlueTeam(e.target.value)}
              >
                <option value="">Select Team</option>
                {teams.map((team) => (
                  <option key={`blue-${team}`} value={team}>
                    {team}
                  </option>
                ))}
              </select>

              <div className="mt-6 min-h-16 border-l-4 border-[#00d1ff] bg-[#00d1ff]/10 p-4">
                <p className="text-xs uppercase tracking-[0.25em] text-slate-400">
                  Locked Selection
                </p>
                <p className="text-2xl font-bold uppercase tracking-widest text-[#00d1ff]">
                  {blueTeam || "Awaiting Lock"}
                </p>
              </div>
            </motion.div>
          </div>

          <motion.div
            animate={
              blueTeam && redTeam
                ? {
                    scale: [1, 1.08, 1],
                    textShadow: [
                      "0 0 8px rgba(245,197,66,0.6)",
                      "0 0 18px rgba(245,197,66,0.9)",
                      "0 0 8px rgba(245,197,66,0.6)",
                    ],
                  }
                : { scale: 1 }
            }
            transition={{
              duration: 1.4,
              repeat: blueTeam && redTeam ? Infinity : 0,
            }}
            className={`text-center text-6xl font-bold uppercase tracking-widest md:text-8xl ${
              blueTeam && redTeam
                ? "text-[#f5c542] drop-shadow-[0_0_18px_rgba(245,197,66,0.9)]"
                : "text-zinc-800"
            }`}
          >
            VS
          </motion.div>

          <div className="relative">
            {redTeam && (
              <div className="absolute inset-[-35px] -z-10 bg-[#ff003c]/35 blur-3xl" />
            )}

            <motion.div
              initial={{ opacity: 0, x: 60 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.45}}
              whileHover={{ scale: 1.02 }}
              className={`
                red-cyber-panel
                border-2
                border-[#ff003c]
                bg-zinc-950
                p-6
                transition-all
                duration-500
                ${
                  redTeam
                    ? "shadow-[0_0_60px_rgba(255,0,60,0.28)]"
                    : "shadow-[0_0_18px_rgba(255,0,60,0.10)]"
                }
              `}
              >
                <p className="mb-4 text-sm font-bold uppercase tracking-[0.3em] text-[#ff003c]">
                  Red Team
                </p>

              <select
                className="w-full border border-[#ff003c]/70 bg-black p-4 text-xl font-bold uppercase tracking-widest text-white outline-none shadow-[inset_0_0_16px_rgba(255,0,60,0.25)]"
                value={redTeam}
                onChange={(e) => setRedTeam(e.target.value)}
              >
                <option value="">Select Team</option>
                {teams.map((team) => (
                  <option key={`red-${team}`} value={team}>
                    {team}
                  </option>
                ))}
              </select>

              <div className="mt-6 min-h-16 border-l-4 border-[#ff003c] bg-[#ff003c]/10 p-4">
                <p className="text-xs uppercase tracking-[0.25em] text-slate-400">
                  Locked Selection
                </p>
                <p className="text-2xl font-bold uppercase tracking-widest text-[#ff003c]">
                  {redTeam || "Awaiting Lock"}
                </p>
              </div>
            </motion.div>
          </div>
        </div>

        <motion.button
          whileHover={!blueTeam || !redTeam || isLoading ? {} : { scale: 1.03 }}
          whileTap={!blueTeam || !redTeam || isLoading ? {} : { scale: 0.97 }}
          onClick={handlePredict}
          disabled={!blueTeam || !redTeam || isLoading}
          className="cyber-panel mx-auto mt-10 w-full max-w-xl bg-[#f5c542] px-8 py-4 text-2xl font-bold uppercase tracking-[0.25em] text-black shadow-[0_0_20px_rgba(245,197,66,0.6)] transition hover:scale-[1.02] hover:bg-white disabled:cursor-not-allowed disabled:bg-zinc-800 disabled:text-zinc-500 disabled:shadow-none"
        >
          {isLoading ? "Calculating" : "Lock In"}
        </motion.button>

        <AnimatePresence>
          {result && (
            <motion.section
              initial={{ opacity: 0, y: 40, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 20, scale: 0.96 }}
              transition={{ duration: 0.35 }}
              className="cyber-panel mt-10 border border-zinc-700 bg-zinc-950 p-6 shadow-[0_0_25px_rgba(255,255,255,0.12)]"
            >
                <div className="mb-5 flex items-center justify-between gap-4">
                  <h2 className="text-3xl font-bold uppercase tracking-[0.2em]">
                    Prediction Locked
                  </h2>

                  <p className="text-xl font-bold uppercase tracking-[0.2em] text-[#f5c542]">
                    {result.predicted_winner} Team Wins
                  </p>
                </div>

                <div className="h-12 w-full overflow-hidden border border-zinc-700 bg-black">
                  <div className="flex h-full w-full">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${bluePct}%` }}
                      transition={{ duration: 0.8, ease: "easeOut" }}
                      className="flex items-center justify-start bg-[#00d1ff] pl-4 font-bold uppercase tracking-widest text-black transition-all duration-700"
                      style={{ width: `${bluePct}%` }}
                    >
                      {bluePct.toFixed(1)}%
                    </motion.div>

                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${redPct}%` }}
                      transition={{ duration: 0.8, ease: "easeOut", delay: 0.2 }}
                      className="flex items-center justify-end bg-[#ff003c] pr-4 font-bold uppercase tracking-widest text-white transition-all duration-700"
                      style={{ width: `${redPct}%` }}
                    >
                      {redPct.toFixed(1)}%
                    </motion.div>
                  </div>
                </div>

                <div className="mt-4 flex justify-between text-sm font-bold uppercase tracking-[0.2em] text-slate-400">
                  <span>{result.blue_team}</span>
                  <span>{result.red_team}</span>
                </div>
            </motion.section>
          )}
        </AnimatePresence>
      </section>
    </main>
  );
}

export default App;