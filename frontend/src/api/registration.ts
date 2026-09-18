import { api } from "./client";
import type {
  RegisterStartResponse,
  RegistrationStatusResponse,
  RunSummary,
  RegistrationImagesResponse,
  ComparisonResponse,
} from "../types";

export interface RegisterOptions {
  sensors?: string[];
  force_rerun?: boolean;
  max_keypoints?: number;
  ratio_thresh?: number;
  min_matches?: number;
  reproj_thresh?: number;
}

export const registrationApi = {
  start: (options: RegisterOptions = {}) => api.post<RegisterStartResponse>("/api/register", options),
  status: (runId: string) => api.get<RegistrationStatusResponse>(`/api/register/${runId}/status`),
  results: (runId: string) => api.get<RunSummary>(`/api/register/${runId}/results`),
  metrics: (runId: string) => api.get<Record<string, unknown>>(`/api/register/${runId}/metrics`),
  matches: (runId: string) => api.get<{ images: string[] }>(`/api/register/${runId}/matches`),
  residuals: (runId: string) => api.get<Record<string, string[]>>(`/api/register/${runId}/residuals`),
  images: (runId: string) => api.get<RegistrationImagesResponse>(`/api/register/${runId}/images`),
  comparison: (runId: string) => api.get<ComparisonResponse>(`/api/register/${runId}/comparison`),
  output: (runId: string) => api.get<Record<string, unknown>>(`/api/output?run_id=${encodeURIComponent(runId)}`),
};
