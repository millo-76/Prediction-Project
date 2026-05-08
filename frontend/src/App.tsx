import { useEffect, useState } from "react";
import { getTeams, predictMatch } from "./api/predictorApi";

function App() {
  const [teams, setTeams] = useState<string[]>([]);
  const [blueTeam, setBlueTeam] = useState("");
  const [redTeam, setRedTeam] = useState("");
  const [result, setResult] = useState<any>(null);

  useEffect(() => {
    async function loadTeams() {
      const teamList = await getTeams();
      setTeams(teamList);
    }

    loadTeams();
  }, []);

  async function handlePredict() {
    if (!blueTeam || !redTeam) return;

    const data = await predictMatch(blueTeam, redTeam);
    setResult(data);
  }

  return (
    <main className="min-h-screen bg-slate-950 text-white px-6 py-10">
      <section className="mx-auto max-w-3xl rounded-2xl border border-slate-800 bg-slate-900 p-8 shadow-xl">

        <h1 className="mb-8 text-4xl font-bold">
          League of Legends Match Predictor
        </h1>

        <div className="grid gap-4 md:grid-cols-2">
          <select
            className="rounded-xl border border-blue-500/40 bg-slate-950 p-3 text-white"
            value={blueTeam}
            onChange={(e) => setBlueTeam(e.target.value)}
          >
            <option value="">Select Blue Team</option>
            {teams.map((team) => (
              <option key={`blue-${team}`} value={team}>
                {team}
              </option>
            ))}
          </select>

          <select
            className="rounded-xl border border-red-500/40 bg-slate-950 p-3 text-white"
            value={redTeam}
            onChange={(e) => setRedTeam(e.target.value)}
          >
            <option value="">Select Red Team</option>
            {teams.map((team) => (
              <option key={`red-${team}`} value={team}>
                {team}
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={handlePredict}
          className="mt-6 w-full rounded-xl bg-indigo-600 px-4 py-3 font-semibold transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-700"
          disabled={!blueTeam || !redTeam}
        >
          Predict Match
        </button>

        {result && (
          <section className="mt-8 rounded-2xl border border-slate-800 bg-slate-950 p-6">
            <h2 className="mb-4 text-2xl font-bold">Prediction Result</h2>

            <p className="mb-4 text-lg">
              Predicted Winner:{" "}
              <span className="font-bold text-indigo-400">
                {result.predicted_winner}
              </span>
            </p>

            <div className="space-y-4">
              <div>
                <div className="mb-1 flex justify-between text-sm">
                  <span>Blue Win Probability</span>
                  <span>{(result.blue_win_probability * 100).toFixed(2)}%</span>
                </div>
                <div className="h-3 rounded-full bg-slate-800">
                  <div
                    className="h-3 rounded-full bg-blue-500"
                    style={{
                      width: `${result.blue_win_probability * 100}%`,
                    }}
                  />
                </div>
              </div>

              <div>
                <div className="mb-1 flex justify-between text-sm">
                  <span>Red Win Probability</span>
                  <span>{(result.red_win_probability * 100).toFixed(2)}%</span>
                </div>
                <div className="h-3 rounded-full bg-slate-800">
                  <div
                    className="h-3 rounded-full bg-red-500"
                    style={{
                      width: `${result.red_win_probability * 100}%`,
                    }}
                  />
                </div>
              </div>
            </div>
          </section>
        )}
      </section>
    </main>
  );
}

export default App;