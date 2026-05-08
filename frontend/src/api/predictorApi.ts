import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000";

export async function getTeams() {
  const response = await axios.get(`${API_BASE_URL}/teams`);
  return response.data.teams;
}

export async function predictMatch(blueTeam: string, redTeam: string) {
  const response = await axios.post(`${API_BASE_URL}/predict-from-teams`, {
    blue_team: blueTeam,
    red_team: redTeam,
  });

  return response.data;
}