using System.Globalization;
using System.Text;
using System.IO.MemoryMappedFiles;

using PyTorchCheckpoint;


const int ExitOk = 0;
const int ExitBadArgs = 2;
const int ExitNonFinite = 3;


try {
    CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
    CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

    var parsed = Args.Parse(args);
    if (parsed.Error is not null) {
        Console.Error.WriteLine(parsed.Error);
        PrintUsage();
        return ExitBadArgs;
    }

    if (parsed.ShowHelp) {
        PrintUsage();
        return ExitOk;
    }

    string? path = ResolveInputPath(parsed.Path);
    if (path is null) {
        Console.Error.WriteLine("Missing checkpoint path.");
        PrintUsage();
        return ExitBadArgs;
    }

    if (!File.Exists(path)) {
        Console.Error.WriteLine($"File not found: {path}");
        return ExitBadArgs;
    }

    using var checkpoint = TorchCheckpoint.Open(path);

    switch (parsed.Command) {
    case Command.CheckFinite:
        return RunCheckFinite(checkpoint, path, parsed);
    case Command.List:
        return RunList(checkpoint, path, parsed);
    default:
        Console.Error.WriteLine("Missing command.");
        PrintUsage();
        return ExitBadArgs;
    }
} catch (Exception ex) {
    Console.Error.WriteLine(ex.ToString());
    return 1;
}

static int RunCheckFinite(TorchCheckpoint checkpoint, string path, Args parsed) {
    Console.WriteLine($"Loaded checkpoint: {path}");
    Console.WriteLine($"Tensors: {checkpoint.Tensors.Count}");

    var floatTensorsByStorageKey = new Dictionary<string, List<TorchTensor>>(StringComparer.Ordinal);
    foreach (var tensor in checkpoint.Tensors) {
        if (!tensor.Storage.ScalarType.IsFloatingPoint()) {
            continue;
        }

        if (!floatTensorsByStorageKey.TryGetValue(tensor.Storage.Key, out var list)) {
            list = [];
            floatTensorsByStorageKey[tensor.Storage.Key] = list;
        }
        list.Add(tensor);
    }

    int nonFiniteTensors = 0;
    long checkedElements = 0;

    foreach (var kvp in floatTensorsByStorageKey) {
        var tensors = kvp.Value;
        var storage = tensors[0].Storage;

        for (int i = 0; i < tensors.Count; i++) {
            TensorScanner.ValidateTensorStorageRange(tensors[i]);
        }

        using var storageData = StorageData.Open(checkpoint, storage, parsed.UseMemoryMappedFiles);
        foreach (var tensor in tensors) {
            bool ok = TensorScanner.CheckFinite(storageData, checkpoint.IsLittleEndian, tensor, out var hit, out long elementsChecked);
            checkedElements += elementsChecked;
            if (!ok) {
                nonFiniteTensors++;
                Console.Error.WriteLine($"{tensor.Name ?? "<unnamed>"}: non-finite {hit.Kind} at linearIndex={hit.LinearIndex} storageKey={tensor.Storage.Key} storageIndex={hit.StorageIndex}");
            }
        }
    }

    if (nonFiniteTensors > 0) {
        Console.Error.WriteLine($"FAILED: {nonFiniteTensors} tensor(s) contain NaN/Inf. Checked {checkedElements} element(s).");
        return ExitNonFinite;
    }

    Console.WriteLine($"OK: all floating-point tensors are finite. Checked {checkedElements} element(s).");
    return ExitOk;
}

static int RunList(TorchCheckpoint checkpoint, string path, Args parsed) {
    Console.WriteLine($"Loaded checkpoint: {path}");
    Console.WriteLine($"Tensors: {checkpoint.Tensors.Count}");

    var rows = new List<ListRow>(checkpoint.Tensors.Count);
    var rowsByStorageKey = new Dictionary<string, List<int>>(StringComparer.Ordinal);

    for (int i = 0; i < checkpoint.Tensors.Count; i++) {
        TorchTensor tensor = checkpoint.Tensors[i];
        var row = new ListRow(tensor) {
            Name = tensor.Name ?? "<unnamed>",
            DType = tensor.Storage.ScalarType.ToDisplayString(),
            Shape = FormatShape(tensor.Sizes),
        };
        rows.Add(row);

        if (!rowsByStorageKey.TryGetValue(tensor.Storage.Key, out var list)) {
            list = [];
            rowsByStorageKey[tensor.Storage.Key] = list;
        }
        list.Add(i);
    }

    int nameWidth = MaxLen(rows, static r => r.Name) + 1;
    int dtypeWidth = Math.Max(MaxLen(rows, static r => r.DType), "dtype".Length) + 1;
    int shapeWidth = Math.Max(MaxLen(rows, static r => r.Shape), "shape".Length) + 1;

    const int NumericContentWidth = 9;
    const int NumericColumnWidth = 10;

    Console.WriteLine(FormatHeader(nameWidth, dtypeWidth, shapeWidth, NumericColumnWidth));

    foreach (var kvp in rowsByStorageKey) {
        List<int> rowIndices = kvp.Value;
        TorchStorage storage = rows[rowIndices[0]].Tensor.Storage;

        for (int i = 0; i < rowIndices.Count; i++) {
            TensorScanner.ValidateTensorStorageRange(rows[rowIndices[i]].Tensor);
        }

        using var storageData = StorageData.Open(checkpoint, storage, parsed.UseMemoryMappedFiles);
        for (int i = 0; i < rowIndices.Count; i++) {
            ListRow row = rows[rowIndices[i]];
            row.Stats = TensorScanner.ComputeStats(storageData, checkpoint.IsLittleEndian, row.Tensor);
        }
    }

    var sb = new StringBuilder(capacity: 256);
    for (int i = 0; i < rows.Count; i++) {
        sb.Clear();
        ListRow row = rows[i];

        AppendTextColumn(sb, row.Name, nameWidth);
        AppendTextColumn(sb, row.DType, dtypeWidth);
        AppendTextColumn(sb, row.Shape, shapeWidth);

        if (row.Tensor.Numel == 1) {
            AppendNumberColumn(sb, row.Stats.Mean, NumericContentWidth, NumericColumnWidth);
            AppendBlankColumn(sb, NumericColumnWidth);
            AppendBlankColumn(sb, NumericColumnWidth);
            AppendBlankColumn(sb, NumericColumnWidth);
            AppendBlankColumn(sb, NumericColumnWidth);
        } else {
            AppendNumberColumn(sb, row.Stats.Mean, NumericContentWidth, NumericColumnWidth);
            AppendNumberColumn(sb, row.Stats.Std, NumericContentWidth, NumericColumnWidth);
            AppendNumberColumn(sb, row.Stats.Min, NumericContentWidth, NumericColumnWidth);
            AppendNumberColumn(sb, row.Stats.Max, NumericContentWidth, NumericColumnWidth);
            AppendNumberColumn(sb, row.Stats.MinAbs, NumericContentWidth, NumericColumnWidth);
        }

        Console.WriteLine(sb.ToString());
    }

    return ExitOk;
}

static string? ResolveInputPath(string? argPath) {
    if (!string.IsNullOrWhiteSpace(argPath)) {
        return argPath;
    }

    string[] pts = Directory.GetFiles(Environment.CurrentDirectory, "*.pt", SearchOption.TopDirectoryOnly);
    return pts.Length == 1 ? pts[0] : null;
}

static string FormatShape(IReadOnlyList<long> sizes) {
    if (sizes.Count == 0) return "[]";
    string[] parts = new string[sizes.Count];
    for (int i = 0; i < sizes.Count; i++) parts[i] = sizes[i].ToString(CultureInfo.InvariantCulture);
    return "[" + string.Join(",", parts) + "]";
}

static string FormatHeader(int nameWidth, int dtypeWidth, int shapeWidth, int numericWidth) {
    var sb = new StringBuilder(capacity: 128);

    AppendTextColumn(sb, "name", nameWidth);
    AppendTextColumn(sb, "dtype", dtypeWidth);
    AppendTextColumn(sb, "shape", shapeWidth);

    AppendTextColumn(sb, "mean", numericWidth, rightAlign: true);
    AppendTextColumn(sb, "std", numericWidth, rightAlign: true);
    AppendTextColumn(sb, "min", numericWidth, rightAlign: true);
    AppendTextColumn(sb, "max", numericWidth, rightAlign: true);
    AppendTextColumn(sb, "minabs", numericWidth, rightAlign: true);

    return sb.ToString();
}

static void AppendBlankColumn(StringBuilder sb, int columnWidth) {
    if (columnWidth < 1) throw new ArgumentOutOfRangeException(nameof(columnWidth));
    sb.Append(' ', columnWidth);
}

static void AppendTextColumn(StringBuilder sb, string text, int width, bool rightAlign = false) {
    if (width < 1) throw new ArgumentOutOfRangeException(nameof(width));

    string value = text ?? "";
    if (value.Length > width - 1) {
        value = value.Substring(0, width - 1);
    }

    if (rightAlign) {
        sb.Append(value.PadLeft(width - 1));
    } else {
        sb.Append(value.PadRight(width - 1));
    }
    sb.Append(' ');
}

static void AppendNumberColumn(StringBuilder sb, double value, int contentWidth, int columnWidth) {
    if (contentWidth < 1) throw new ArgumentOutOfRangeException(nameof(contentWidth));
    if (columnWidth < contentWidth + 1) throw new ArgumentOutOfRangeException(nameof(columnWidth));

    string s = FormatNumber(value, contentWidth);
    if (s.Length > contentWidth) s = OverflowPlaceholder(contentWidth);

    sb.Append(s.PadLeft(contentWidth));
    sb.Append(' ', columnWidth - contentWidth);
}

static string FormatNumber(double value, int maxWidth) {
    if (maxWidth < 1) throw new ArgumentOutOfRangeException(nameof(maxWidth));

    if (double.IsNaN(value)) return maxWidth >= 3 ? "nan" : OverflowPlaceholder(maxWidth);
    if (double.IsPositiveInfinity(value)) return maxWidth >= 4 ? "+inf" : OverflowPlaceholder(maxWidth);
    if (double.IsNegativeInfinity(value)) return maxWidth >= 4 ? "-inf" : OverflowPlaceholder(maxWidth);
    if (value == 0) return "0";

    double abs = Math.Abs(value);

    bool preferScientific = abs >= 1e6 || abs < 1e-3;
    if (!preferScientific) {
        string fixedS = FormatFixedForWidth(value, maxWidth, maxFractionDigits: 4);
        if (fixedS.Length <= maxWidth) return fixedS;
    }

    string sciS = FormatScientificForWidth(value, maxWidth, maxMantissaDecimals: 3);
    if (sciS.Length <= maxWidth) return sciS;

    string fixedFallback = FormatFixedForWidth(value, maxWidth, maxFractionDigits: 0);
    if (fixedFallback.Length <= maxWidth) return fixedFallback;

    return OverflowPlaceholder(maxWidth);
}

static string FormatFixedForWidth(double value, int maxWidth, int maxFractionDigits) {
    if (maxWidth < 1) return OverflowPlaceholder(maxWidth);

    string sign = value < 0 ? "-" : "";
    double abs = Math.Abs(value);

    int intDigits;
    if (abs >= 1) {
        intDigits = (int)Math.Floor(Math.Log10(abs)) + 1;
    } else {
        intDigits = 1; // "0"
    }

    int baseLen = sign.Length + intDigits;
    if (baseLen > maxWidth) {
        return OverflowPlaceholder(maxWidth);
    }

    int decimalsByWidth = maxWidth - baseLen - 1; // minus '.'
    if (decimalsByWidth <= 0) {
        return value.ToString("0", CultureInfo.InvariantCulture);
    }

    int decimals = Math.Min(maxFractionDigits, decimalsByWidth);
    string fmt = "0." + new string('#', decimals);
    return value.ToString(fmt, CultureInfo.InvariantCulture);
}

static string FormatScientificForWidth(double value, int maxWidth, int maxMantissaDecimals) {
    if (maxWidth < 1) return OverflowPlaceholder(maxWidth);

    string sign = value < 0 ? "-" : "";
    double abs = Math.Abs(value);
    if (abs == 0) return "0";

    int exp10 = (int)Math.Floor(Math.Log10(abs));
    double mantissa = abs / Math.Pow(10, exp10);

    string expStr = exp10 >= 0 ? "+" + exp10.ToString(CultureInfo.InvariantCulture) : exp10.ToString(CultureInfo.InvariantCulture);

    int baseLen = sign.Length + 1 /*digit*/ + 1 /*E*/ + expStr.Length;
    if (baseLen > maxWidth) return OverflowPlaceholder(maxWidth);

    int decimalsByWidth = maxWidth - baseLen - 1; // '.'
    int decimals = Math.Max(0, Math.Min(maxMantissaDecimals, decimalsByWidth));

    string mantissaFmt = decimals == 0 ? "0" : "0." + new string('#', decimals);
    string mantissaStr = mantissa.ToString(mantissaFmt, CultureInfo.InvariantCulture);
    return sign + mantissaStr + "E" + expStr;
}

static string OverflowPlaceholder(int width) => new('*', width);

static int MaxLen(List<ListRow> rows, Func<ListRow, string> selector) {
    int max = 0;
    for (int i = 0; i < rows.Count; i++) {
        int len = selector(rows[i]).Length;
        if (len > max) max = len;
    }
    return max;
}

static void PrintUsage() {
    Console.WriteLine("ptt check finite [--mmap] <checkpoint.pt>");
    Console.WriteLine("ptt ls          [--mmap] <checkpoint.pt>");
    Console.WriteLine();
    Console.WriteLine("Commands:");
    Console.WriteLine("  check finite   Fail if any floating-point tensor contains NaN or +/-Inf.");
    Console.WriteLine("  ls             List tensors with shape, dtype, and basic stats (min/max/min(abs)/mean/std).");
    Console.WriteLine();
    Console.WriteLine("Options:");
    Console.WriteLine("  --mmap         Materialize each storage to a temp file and use MemoryMappedFile for access.");
    Console.WriteLine();
    Console.WriteLine("If <checkpoint.pt> is omitted and there is exactly one *.pt file in the current directory, it is used.");
}

internal enum Command {
    Unknown = 0,
    CheckFinite,
    List,
}

internal sealed class ListRow {
    public ListRow(TorchTensor tensor) {
        this.Tensor = tensor ?? throw new ArgumentNullException(nameof(tensor));
    }

    public TorchTensor Tensor { get; }
    public string Name { get; init; } = "";
    public string DType { get; init; } = "";
    public string Shape { get; init; } = "";
    public TensorStats Stats { get; set; }
}

internal sealed class Args {
    public Command Command { get; init; }
    public string? Path { get; init; }
    public bool UseMemoryMappedFiles { get; init; }
    public bool ShowHelp { get; init; }
    public string? Error { get; init; }

    public static Args Parse(string[] args) {
        if (args.Length == 0) {
            return new Args { ShowHelp = true };
        }

        bool showHelp = false;
        bool mmap = false;
        var positional = new List<string>();

        for (int i = 0; i < args.Length; i++) {
            string a = args[i];
            if (a == "--help" || a == "-h" || a == "/?") {
                showHelp = true;
                continue;
            }

            if (a == "--mmap") {
                mmap = true;
                continue;
            }

            positional.Add(a);
        }

        if (showHelp) {
            return new Args { ShowHelp = true };
        }

        if (positional.Count == 0) {
            return new Args { ShowHelp = true };
        }

        Command cmd = Command.Unknown;
        int idx = 0;

        if (positional.Count >= 2 && positional[0] == "check" && positional[1] == "finite") {
            cmd = Command.CheckFinite;
            idx = 2;
        } else if (positional[0] == "ls") {
            cmd = Command.List;
            idx = 1;
        }

        if (cmd == Command.Unknown) {
            return new Args { Error = "Unknown command.", ShowHelp = true };
        }

        string? path = idx < positional.Count ? positional[idx] : null;
        if (idx + 1 < positional.Count) {
            return new Args { Error = "Too many arguments.", ShowHelp = true };
        }

        return new Args {
            Command = cmd,
            Path = path,
            UseMemoryMappedFiles = mmap,
        };
    }
}

internal sealed class StorageData: IDisposable {
    readonly byte[]? bytes;
    readonly MemoryMappedFile? mmf;
    readonly MemoryMappedViewAccessor? accessor;
    readonly string? tempPath;
    bool isDisposed;

    StorageData(byte[] bytes) {
        this.bytes = bytes ?? throw new ArgumentNullException(nameof(bytes));
        this.LengthBytes = bytes.LongLength;
    }

    StorageData(MemoryMappedFile mmf, MemoryMappedViewAccessor accessor, string tempPath, long lengthBytes) {
        this.mmf = mmf ?? throw new ArgumentNullException(nameof(mmf));
        this.accessor = accessor ?? throw new ArgumentNullException(nameof(accessor));
        this.tempPath = tempPath ?? throw new ArgumentNullException(nameof(tempPath));
        this.LengthBytes = lengthBytes;
    }

    public long LengthBytes { get; }

    public static StorageData Open(TorchCheckpoint checkpoint, TorchStorage storage, bool preferMmap) {
        if (checkpoint is null) throw new ArgumentNullException(nameof(checkpoint));
        if (storage is null) throw new ArgumentNullException(nameof(storage));

        if (!preferMmap) {
            try {
                return new StorageData(checkpoint.GetStorageBytes(storage));
            } catch (InvalidOperationException) {
            }
        }

        string tempPath = Path.Combine(Path.GetTempPath(), $"ptt-storage-{Guid.NewGuid():N}.bin");
        using (var input = checkpoint.OpenStorage(storage))
        using (var output = new FileStream(tempPath, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.Read, bufferSize: 1 << 20)) {
            input.CopyTo(output);
            output.Flush();
        }

        long length = new FileInfo(tempPath).Length;
        var mmf = MemoryMappedFile.CreateFromFile(tempPath, FileMode.Open, mapName: null, capacity: 0, access: MemoryMappedFileAccess.Read);
        var view = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        return new StorageData(mmf, view, tempPath, length);
    }

    public ushort ReadUInt16(long byteOffset, bool littleEndian) {
        if (this.bytes is not null) {
            int o = checked((int)byteOffset);
            return littleEndian
                ? (ushort)(this.bytes[o] | (this.bytes[o + 1] << 8))
                : (ushort)((this.bytes[o] << 8) | this.bytes[o + 1]);
        }

        if (this.accessor is null) throw new ObjectDisposedException(nameof(StorageData));
        short raw = this.accessor.ReadInt16(byteOffset);
        ushort v = unchecked((ushort)raw);
        if (littleEndian != BitConverter.IsLittleEndian) v = Endian.Swap16(v);
        return v;
    }

    public uint ReadUInt32(long byteOffset, bool littleEndian) {
        if (this.bytes is not null) {
            int o = checked((int)byteOffset);
            unchecked {
                return littleEndian
                    ? (uint)(this.bytes[o] | (this.bytes[o + 1] << 8) | (this.bytes[o + 2] << 16) | (this.bytes[o + 3] << 24))
                    : (uint)((this.bytes[o] << 24) | (this.bytes[o + 1] << 16) | (this.bytes[o + 2] << 8) | this.bytes[o + 3]);
            }
        }

        if (this.accessor is null) throw new ObjectDisposedException(nameof(StorageData));
        int raw = this.accessor.ReadInt32(byteOffset);
        uint v = unchecked((uint)raw);
        if (littleEndian != BitConverter.IsLittleEndian) v = Endian.Swap32(v);
        return v;
    }

    public ulong ReadUInt64(long byteOffset, bool littleEndian) {
        if (this.bytes is not null) {
            int o = checked((int)byteOffset);
            unchecked {
                if (littleEndian) {
                    return (ulong)this.bytes[o]
                        | ((ulong)this.bytes[o + 1] << 8)
                        | ((ulong)this.bytes[o + 2] << 16)
                        | ((ulong)this.bytes[o + 3] << 24)
                        | ((ulong)this.bytes[o + 4] << 32)
                        | ((ulong)this.bytes[o + 5] << 40)
                        | ((ulong)this.bytes[o + 6] << 48)
                        | ((ulong)this.bytes[o + 7] << 56);
                } else {
                    return ((ulong)this.bytes[o] << 56)
                        | ((ulong)this.bytes[o + 1] << 48)
                        | ((ulong)this.bytes[o + 2] << 40)
                        | ((ulong)this.bytes[o + 3] << 32)
                        | ((ulong)this.bytes[o + 4] << 24)
                        | ((ulong)this.bytes[o + 5] << 16)
                        | ((ulong)this.bytes[o + 6] << 8)
                        | this.bytes[o + 7];
                }
            }
        }

        if (this.accessor is null) throw new ObjectDisposedException(nameof(StorageData));
        long raw = this.accessor.ReadInt64(byteOffset);
        ulong v = unchecked((ulong)raw);
        if (littleEndian != BitConverter.IsLittleEndian) v = Endian.Swap64(v);
        return v;
    }

    public byte ReadByte(long byteOffset) {
        if (this.bytes is not null) {
            return this.bytes[checked((int)byteOffset)];
        }

        if (this.accessor is null) throw new ObjectDisposedException(nameof(StorageData));
        return this.accessor.ReadByte(byteOffset);
    }

    public void Dispose() {
        if (this.isDisposed) return;
        this.isDisposed = true;

        try { this.accessor?.Dispose(); } catch { }
        try { this.mmf?.Dispose(); } catch { }

        if (this.tempPath is not null) {
            try { File.Delete(this.tempPath); } catch { }
        }
    }
}

internal static class TensorScanner {
    internal readonly struct NonFiniteHit {
        public NonFiniteHit(long linearIndex, long storageIndex, string kind) {
            this.LinearIndex = linearIndex;
            this.StorageIndex = storageIndex;
            this.Kind = kind ?? throw new ArgumentNullException(nameof(kind));
        }

        public long LinearIndex { get; }
        public long StorageIndex { get; }
        public string Kind { get; }
    }

    public static bool CheckFinite(StorageData storage, bool littleEndian, TorchTensor tensor, out NonFiniteHit hit, out long elementsChecked) {
        if (storage is null) throw new ArgumentNullException(nameof(storage));
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));

        hit = default;
        elementsChecked = 0;

        if (tensor.Numel == 0) return true;

        int elementSize = tensor.Storage.ScalarType.GetElementSizeBytes();
        if (elementSize <= 0) return true;

        ValidateTensorStorageRange(tensor);

        if (IsPositiveContiguous(tensor.Sizes, tensor.Strides)) {
            long start = tensor.StorageOffsetElements;

            if (start < 0 || tensor.Numel < 0 || start > long.MaxValue - tensor.Numel) {
                throw new InvalidOperationException("Tensor storage range exceeds valid bounds.");
            }

            long endExclusive = start + tensor.Numel;
            for (long i = start; i < endExclusive; i++) {
                long byteOffset = checked(i * (long)elementSize);
                if (!TryIsFiniteAt(storage, byteOffset, littleEndian, tensor.Storage.ScalarType, out string? kind)) {
                    hit = new NonFiniteHit(linearIndex: i - start, storageIndex: i, kind: kind);
                    elementsChecked = i - start + 1;
                    return false;
                }
            }

            elementsChecked = tensor.Numel;
            return true;
        }

        long[] sizes = ToLongArray(tensor.Sizes);
        long[] strides = ToLongArray(tensor.Strides);
        long[] indices = new long[sizes.Length];
        long linear = 0;

        while (true) {
            long storageIndex = tensor.StorageOffsetElements;
            for (int d = 0; d < sizes.Length; d++) {
                storageIndex = checked(storageIndex + checked(indices[d] * strides[d]));
            }

            long byteOffset = checked(storageIndex * (long)elementSize);
            if (!TryIsFiniteAt(storage, byteOffset, littleEndian, tensor.Storage.ScalarType, out string? kind)) {
                hit = new NonFiniteHit(linearIndex: linear, storageIndex: storageIndex, kind: kind);
                elementsChecked = linear + 1;
                return false;
            }

            linear++;
            elementsChecked = linear;

            int dim = sizes.Length - 1;
            for (; dim >= 0; dim--) {
                indices[dim]++;
                if (indices[dim] < sizes[dim]) break;
                indices[dim] = 0;
            }

            if (dim < 0) break;
        }

        return true;
    }

    public static TensorStats ComputeStats(StorageData storage, bool littleEndian, TorchTensor tensor) {
        if (storage is null) throw new ArgumentNullException(nameof(storage));
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));

        int elementSize = tensor.Storage.ScalarType.GetElementSizeBytes();
        if (elementSize <= 0 || tensor.Numel == 0) {
            return TensorStats.Empty(tensor.Numel);
        }

        ValidateTensorStorageRange(tensor);

        double min = double.PositiveInfinity;
        double max = double.NegativeInfinity;
        double minAbs = double.PositiveInfinity;
        long nonFinite = 0;

        long count = 0;
        double mean = 0;
        double m2 = 0;

        if (IsPositiveContiguous(tensor.Sizes, tensor.Strides)) {
            long start = tensor.StorageOffsetElements;
            long endExclusive = checked(start + tensor.Numel);
            for (long i = start; i < endExclusive; i++) {
                long byteOffset = checked(i * (long)elementSize);
                if (!TryReadAsDouble(storage, byteOffset, littleEndian, tensor.Storage.ScalarType, out double value, out string? kind)) {
                    nonFinite++;
                    continue;
                }

                if (value < min) min = value;
                if (value > max) max = value;

                double abs = Math.Abs(value);
                if (abs < minAbs) minAbs = abs;

                count++;
                double delta = value - mean;
                mean += delta / count;
                double delta2 = value - mean;
                m2 += delta * delta2;
            }
        } else {
            long[] sizes = ToLongArray(tensor.Sizes);
            long[] strides = ToLongArray(tensor.Strides);
            long[] indices = new long[sizes.Length];

            while (true) {
                long storageIndex = tensor.StorageOffsetElements;
                for (int d = 0; d < sizes.Length; d++) {
                    storageIndex = checked(storageIndex + checked(indices[d] * strides[d]));
                }

                long byteOffset = checked(storageIndex * (long)elementSize);
                if (!TryReadAsDouble(storage, byteOffset, littleEndian, tensor.Storage.ScalarType, out double value, out string? kind)) {
                    nonFinite++;
                } else {
                    if (value < min) min = value;
                    if (value > max) max = value;

                    double abs = Math.Abs(value);
                    if (abs < minAbs) minAbs = abs;

                    count++;
                    double delta = value - mean;
                    mean += delta / count;
                    double delta2 = value - mean;
                    m2 += delta * delta2;
                }

                int dim = sizes.Length - 1;
                for (; dim >= 0; dim--) {
                    indices[dim]++;
                    if (indices[dim] < sizes[dim]) break;
                    indices[dim] = 0;
                }

                if (dim < 0) break;
            }
        }

        if (count == 0) {
            return new TensorStats(double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, tensor.Numel, nonFinite);
        }

        double variance = m2 / count;
        double std = Math.Sqrt(variance);
        return new TensorStats(min, max, minAbs, mean, std, tensor.Numel, nonFinite);
    }

    public static void ValidateTensorStorageRange(TorchTensor tensor) {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));

        int elementSize = tensor.Storage.ScalarType.GetElementSizeBytes();
        if (elementSize <= 0) return;

        var sizes = tensor.Sizes;
        var strides = tensor.Strides;

        if (sizes.Count != strides.Count) {
            throw new InvalidOperationException("Tensor sizes and strides rank mismatch.");
        }

        if (tensor.Numel == 0) return;

        long minIndex = tensor.StorageOffsetElements;
        long maxIndex = tensor.StorageOffsetElements;

        for (int d = 0; d < sizes.Count; d++) {
            long size = sizes[d];
            if (size <= 0) continue;

            long extent;
            checked {
                extent = (size - 1) * strides[d];
            }

            if (extent >= 0) {
                maxIndex = checked(maxIndex + extent);
            } else {
                minIndex = checked(minIndex + extent);
            }
        }

        if (minIndex < 0) {
            throw new InvalidDataException($"Tensor requires negative storage index ({minIndex}).");
        }

        if (maxIndex >= tensor.Storage.ElementCount) {
            throw new InvalidDataException($"Tensor reads past end of storage '{tensor.Storage.Key}': maxIndex={maxIndex}, storageElements={tensor.Storage.ElementCount}.");
        }
    }

    static bool IsPositiveContiguous(IReadOnlyList<long> sizes, IReadOnlyList<long> strides) {
        if (sizes.Count != strides.Count) return false;
        if (sizes.Count == 0) return true;

        long expected = 1;
        for (int d = sizes.Count - 1; d >= 0; d--) {
            long size = sizes[d];
            if (size == 1) continue;
            if (strides[d] != expected) return false;
            expected = checked(expected * size);
        }
        return true;
    }

    static long[] ToLongArray(IReadOnlyList<long> values) {
        long[] arr = new long[values.Count];
        for (int i = 0; i < arr.Length; i++) arr[i] = values[i];
        return arr;
    }

    static bool TryIsFiniteAt(StorageData storage, long byteOffset, bool littleEndian, TorchScalarType scalarType, out string kind) {
        kind = "";
        switch (scalarType) {
        case TorchScalarType.Float32: {
                uint bits = storage.ReadUInt32(byteOffset, littleEndian);
                if ((bits & 0x7F800000u) == 0x7F800000u) {
                    kind = (bits & 0x007FFFFFu) == 0 ? "inf" : "nan";
                    return false;
                }
                return true;
            }
        case TorchScalarType.Float64: {
                ulong bits = storage.ReadUInt64(byteOffset, littleEndian);
                if ((bits & 0x7FF0000000000000ul) == 0x7FF0000000000000ul) {
                    kind = (bits & 0x000FFFFFFFFFFFFFul) == 0 ? "inf" : "nan";
                    return false;
                }
                return true;
            }
        case TorchScalarType.Float16: {
                ushort bits = storage.ReadUInt16(byteOffset, littleEndian);
                if ((bits & 0x7C00) == 0x7C00) {
                    kind = (bits & 0x03FF) == 0 ? "inf" : "nan";
                    return false;
                }
                return true;
            }
        case TorchScalarType.BFloat16: {
                ushort bits = storage.ReadUInt16(byteOffset, littleEndian);
                if ((bits & 0x7F80) == 0x7F80) {
                    kind = (bits & 0x007F) == 0 ? "inf" : "nan";
                    return false;
                }
                return true;
            }
        default:
            return true;
        }
    }

    static bool TryReadAsDouble(StorageData storage, long byteOffset, bool littleEndian, TorchScalarType scalarType, out double value, out string kind) {
        kind = "";
        switch (scalarType) {
        case TorchScalarType.Float32: {
                uint bits = storage.ReadUInt32(byteOffset, littleEndian);
                if ((bits & 0x7F800000u) == 0x7F800000u) {
                    kind = (bits & 0x007FFFFFu) == 0 ? "inf" : "nan";
                    value = double.NaN;
                    return false;
                }
                value = BitCast.UInt32BitsToSingle(bits);
                return true;
            }
        case TorchScalarType.Float64: {
                ulong bits = storage.ReadUInt64(byteOffset, littleEndian);
                if ((bits & 0x7FF0000000000000ul) == 0x7FF0000000000000ul) {
                    kind = (bits & 0x000FFFFFFFFFFFFFul) == 0 ? "inf" : "nan";
                    value = double.NaN;
                    return false;
                }
                value = BitCast.UInt64BitsToDouble(bits);
                return true;
            }
        case TorchScalarType.Float16: {
                ushort bits = storage.ReadUInt16(byteOffset, littleEndian);
                if ((bits & 0x7C00) == 0x7C00) {
                    kind = (bits & 0x03FF) == 0 ? "inf" : "nan";
                    value = double.NaN;
                    return false;
                }
                value = Float16.ToDouble(bits);
                return true;
            }
        case TorchScalarType.BFloat16: {
                ushort bits = storage.ReadUInt16(byteOffset, littleEndian);
                if ((bits & 0x7F80) == 0x7F80) {
                    kind = (bits & 0x007F) == 0 ? "inf" : "nan";
                    value = double.NaN;
                    return false;
                }
                value = BitCast.UInt32BitsToSingle((uint)bits << 16);
                return true;
            }
        case TorchScalarType.Int64: {
                long v = unchecked((long)storage.ReadUInt64(byteOffset, littleEndian));
                value = v;
                return true;
            }
        case TorchScalarType.Int32: {
                int v = unchecked((int)storage.ReadUInt32(byteOffset, littleEndian));
                value = v;
                return true;
            }
        case TorchScalarType.Int16: {
                short v = unchecked((short)storage.ReadUInt16(byteOffset, littleEndian));
                value = v;
                return true;
            }
        case TorchScalarType.Int8: {
                sbyte v = unchecked((sbyte)storage.ReadByte(byteOffset));
                value = v;
                return true;
            }
        case TorchScalarType.UInt8: {
                byte v = storage.ReadByte(byteOffset);
                value = v;
                return true;
            }
        case TorchScalarType.Bool: {
                byte v = storage.ReadByte(byteOffset);
                value = v == 0 ? 0 : 1;
                return true;
            }
        default:
            value = double.NaN;
            return false;
        }
    }
}

internal readonly struct TensorStats {
    public TensorStats(double min, double max, double minAbs, double mean, double std, long elementCount, long nonFiniteCount) {
        this.Min = min;
        this.Max = max;
        this.MinAbs = minAbs;
        this.Mean = mean;
        this.Std = std;
        this.ElementCount = elementCount;
        this.NonFiniteCount = nonFiniteCount;
    }

    public double Min { get; }
    public double Max { get; }
    public double MinAbs { get; }
    public double Mean { get; }
    public double Std { get; }
    public long ElementCount { get; }
    public long NonFiniteCount { get; }

    public static TensorStats Empty(long elementCount) => new(double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, elementCount, 0);
}

internal static class Endian {
    public static ushort Swap16(ushort v) => (ushort)((v << 8) | (v >> 8));

    public static uint Swap32(uint v) {
        return (v >> 24)
            | ((v >> 8) & 0x0000FF00u)
            | ((v << 8) & 0x00FF0000u)
            | (v << 24);
    }

    public static ulong Swap64(ulong v) {
        return (v >> 56)
            | ((v >> 40) & 0x000000000000FF00ul)
            | ((v >> 24) & 0x0000000000FF0000ul)
            | ((v >> 8) & 0x00000000FF000000ul)
            | ((v << 8) & 0x000000FF00000000ul)
            | ((v << 24) & 0x0000FF0000000000ul)
            | ((v << 40) & 0x00FF000000000000ul)
            | (v << 56);
    }
}

internal static class BitCast {
    public static unsafe float UInt32BitsToSingle(uint bits) => *(float*)&bits;
    public static unsafe double UInt64BitsToDouble(ulong bits) => *(double*)&bits;
}

internal static class Float16 {
    public static double ToDouble(ushort bits) => ToSingle(bits);

    public static float ToSingle(ushort bits) {
        uint sign = (uint)(bits >> 15) & 0x1u;
        uint exp = (uint)(bits >> 10) & 0x1Fu;
        uint frac = (uint)bits & 0x3FFu;

        if (exp == 0) {
            if (frac == 0) {
                return sign == 0 ? 0f : -0f;
            }

            float mantissa = frac / 1024f;
            float value = (float)Math.Pow(2, -14) * mantissa;
            return sign == 0 ? value : -value;
        }

        if (exp == 31) {
            return frac == 0 ? (sign == 0 ? float.PositiveInfinity : float.NegativeInfinity) : float.NaN;
        }

        float m = 1f + (frac / 1024f);
        int e = (int)exp - 15;
        float result = (float)Math.Pow(2, e) * m;
        return sign == 0 ? result : -result;
    }
}

internal static class TorchScalarTypeCliExtensions {
    public static bool IsFloatingPoint(this TorchScalarType t) {
        return t == TorchScalarType.Float16
            || t == TorchScalarType.BFloat16
            || t == TorchScalarType.Float32
            || t == TorchScalarType.Float64;
    }

    public static string ToDisplayString(this TorchScalarType t) {
        return t switch {
            TorchScalarType.Float16 => "fp16",
            TorchScalarType.BFloat16 => "bf16",
            TorchScalarType.Float32 => "fp32",
            TorchScalarType.Float64 => "fp64",
            TorchScalarType.Int64 => "i64",
            TorchScalarType.Int32 => "i32",
            TorchScalarType.Int16 => "i16",
            TorchScalarType.Int8 => "i8",
            TorchScalarType.UInt8 => "u8",
            TorchScalarType.Bool => "bool",
            _ => "unknown",
        };
    }
}
