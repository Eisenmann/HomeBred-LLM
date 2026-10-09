namespace HomebredLLM.Models;

/// <summary>User's choice of where llama.cpp runs. Auto = GPU when a GPU backend is loaded, otherwise CPU.</summary>
public enum ComputeMode { Auto = 0, Cpu = 1, Gpu = 2 }
