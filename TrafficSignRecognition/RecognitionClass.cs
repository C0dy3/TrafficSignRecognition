using System;
using System.Drawing;
using System.Windows.Forms;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace AutoVisionProjekt
{
    public partial class Form1 : Form
    {
        private Mat _imgOriginal;
        private PictureBox _pb;
        private Label _status;

        public Form1()
        {
            // Nastavení UI aplikace
            this.Size = new System.Drawing.Size(1300, 900);
            this.Text = "ADAS Ultra-Precision Mode 2026";
            this.StartPosition = FormStartPosition.CenterScreen;

            var panelTop = new Panel { Dock = DockStyle.Top, Height = 60, BackColor = Color.FromArgb(45, 45, 48) };

            var btnLoad = new Button
            {
                Text = "1. NAHRÁT SNÍMEK",
                Location = new System.Drawing.Point(15, 10),
                Width = 150, Height = 40,
                FlatStyle = FlatStyle.Flat, ForeColor = Color.White, Font = new Font("Segoe UI", 9, FontStyle.Bold)
            };

            var btnRun = new Button
            {
                Text = "2. ANALÝZA ",
                Location = new System.Drawing.Point(180, 10),
                Width = 180, Height = 40,
                FlatStyle = FlatStyle.Flat, ForeColor = Color.LimeGreen, Font = new Font("Segoe UI", 9, FontStyle.Bold)
            };

            _status = new Label
            {
                Text = "Čekám na obrázek...",
                Location = new System.Drawing.Point(380, 22),
                ForeColor = Color.White, AutoSize = true, Font = new Font("Segoe UI", 10)
            };

            _pb = new PictureBox
            {
                Dock = DockStyle.Fill,
                SizeMode = PictureBoxSizeMode.Zoom,
                BackColor = Color.Black
            };

            btnLoad.Click += (s, e) =>
            {
                using (var ofd = new OpenFileDialog { Filter = "Images|*.jpg;*.png;*.bmp" })
                {
                    if (ofd.ShowDialog() == DialogResult.OK)
                    {
                        _imgOriginal = Cv2.ImRead(ofd.FileName);
                        _pb.Image = BitmapConverter.ToBitmap(_imgOriginal);
                        _status.Text = "Obrázek připraven k hloubkové analýze.";
                    }
                }
            };

            btnRun.Click += BtnRun_Click;

            panelTop.Controls.Add(btnLoad);
            panelTop.Controls.Add(btnRun);
            panelTop.Controls.Add(_status);
            this.Controls.Add(_pb);
            this.Controls.Add(panelTop);
        }

        private void BtnRun_Click(object sender, EventArgs e)
        {
            if (_imgOriginal == null)
            {
                MessageBox.Show("Nejdřív nahraj obrázek!");
                return;
            }

            _status.Text = "Provádím precizní analýzu (CLAHE + Contour Analysis)...";
            Application.DoEvents();

            // Spuštění samotné logiky
            var vysledek = PrecesizeAnalysys(_imgOriginal);

            _pb.Image = BitmapConverter.ToBitmap(vysledek);
            _status.Text = "Analýza dokončena. Všechny objekty identifikovány.";
        }

        // HLAVNÍ LOGIKA DETEKCE 
        //Princip - vytvoří se maska na celém obrázku, na obrázku se díky značkám, které mají vyráznější barvy
        //Vytvoří jakési ostrůvky těch barev, konkrétně modrá nebo červená
        private Mat PrecesizeAnalysys(Mat input)
        {
            var frame = input.Clone();

            // 1. ZLEPŠENÍ OBRAZU (Proti protisvětlu a mlze)
            using (var lab = new Mat())
            {
                Cv2.CvtColor(frame, lab, ColorConversionCodes.BGR2Lab);
                var planes = Cv2.Split(lab);
                using (var clahe = Cv2.CreateCLAHE(3.0, new OpenCvSharp.Size(8, 8)))
                    clahe.Apply(planes[0], planes[0]); //Precizní změna kontrastu obrazu protisvětlo a stíny
                Cv2.Merge(planes, lab);
                Cv2.CvtColor(lab, frame, ColorConversionCodes.Lab2BGR);
            }

            // DETEKCE ZNAČEK
            using (var hsv = new Mat())
            {
                Cv2.CvtColor(frame, hsv, ColorConversionCodes.BGR2HSV);

                // ČERVENÉ ZNAČKY (Zákazy, Výstrahy)
                using (var redMask = GetRobustRedMask(hsv))
                {
                    // Morfologie: Spojí rozbité kusy značky do jednoho tvaru
                    Cv2.MorphologyEx(redMask, redMask, MorphTypes.Close,
                        Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(5, 5)));
                    DetectShapes(frame, redMask, "CERVENA");
                }

                // MODRÉ ZNAČKY (Příkazové, Informační)
                using (var blueMask = GetRobustBlueMask(hsv))
                {
                    Cv2.MorphologyEx(blueMask, blueMask, MorphTypes.Close,
                        Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(5, 5)));
                    DetectShapes(frame, blueMask, "MODRA");
                }
            }

            // 3. DETEKCE PRUHŮ
            DetectLanes(frame);

            return frame;
        }

        // DETECKE TVARŮ
        private void DetectShapes(Mat frame, Mat mask, string barva)
        {
            Cv2.FindContours(mask, out var contours, out _, RetrievalModes.External,
                ContourApproximationModes.ApproxSimple);

            foreach (var cnt in contours)
            {
                var area = Cv2.ContourArea(cnt);
                if (area < 150) continue;

                var rect = Cv2.BoundingRect(cnt);

                // 1. FILTR POZICE
                // Značka nebude v dolních 25 % obrazu (tam je kapota) 
                // a pravděpodobně ani úplně uprostřed dole (tam je silnice)
                // Bude tak cca. v 75% obrazu někde v úrovni očí plus pár procent nahoru
                //if (rect.Y + rect.Height > frame.Rows * 0.55) continue;

                var pomerStran = (double)rect.Width / rect.Height;
                if (pomerStran is < 0.7 or > 2.5) continue;

                // 2. TEST SOLIDARITY (Skutečná plocha vs plocha obdélníku)
                // Skutečná značka zabírá velkou část svého ohraničení (kruh cca 78 %, trojúhelník 50 %)
                var extent = area / (rect.Width * rect.Height);
                if (extent < 0.4) continue;

                // 3. PŘESNÁ IDENTIFIKACE TVARU - hledám počet hran
                var peri = Cv2.ArcLength(cnt, true);
                var approx = Cv2.ApproxPolyDP(cnt, 0.04 * peri, true);

                var text = "OBJEKT";
                var color = barva == "CERVENA" ? Scalar.Red : Scalar.DeepSkyBlue;

                if (barva == "CERVENA")
                {
                    switch (approx.Length)
                    {
                        // Trojúhelník
                        case 3:
                            text = "DEJ PREDNOST";
                            break;
                        // Kruh má více jak 3 hrany, ale ne 4 protože to je kosočtverece 
                        case > 5:
                        {
                            var circularity = (4 * Math.PI * area) / (peri * peri); //Jak moc kulaté to je 
                            if (circularity > 0.7) text = "ZAKAZ / OMEZENI";
                            else continue; // Není to dostatečně kulaté
                            break;
                        }
                        default:
                            text = "IDK CO TU JE";
                            continue; // Divný tvar, nebo můj deketro opět nefunguje
                    }
                }
                else if(barva == "MODRA") // MODRÁ
                {
                    // Modré značky jsou buď kruhy (Příkaz) nebo čtverce (Info)
                    if (approx.Length >= 4) text = "INFO / PRIKAZ";
                    else continue;
                }
                else // Tady to může být jiná barva
                {
                    text = approx.Length switch
                    {
                        4 => "HLAVNÍ SILNICE",
                        _ => "IDK CO TU JE"
                    };
                }

                // VYZANČENÍ VÝSLEDKŮ HLEDÁNÍ - Takové to hezké ohraničení výsledku
                Cv2.Rectangle(frame, rect, color, 3);
                
                Cv2.PutText(frame, text, new OpenCvSharp.Point(rect.X, rect.Y - 10),
                    HersheyFonts.HersheyComplexSmall, 0.8, Scalar.Black, 3);
                Cv2.PutText(frame, text, new OpenCvSharp.Point(rect.X, rect.Y - 10),
                    HersheyFonts.HersheyComplexSmall, 0.8, color, 1);
            }
        }

        private void DetectLanes(Mat frame)
        {
            // Definujeme oblast zájmu - je to spodek od kapoty, více nahoru moc pruhů nebude
            var startY = (int)(frame.Rows * 0.55);
            var endY = (int)(frame.Rows * 0.88);
            var roiRect = new Rect(0, startY, frame.Cols, endY - startY);

            using var roi = new Mat(frame, roiRect);
            using var hls = new Mat();
            using var binary = new Mat();
            using var edges = new Mat();
            //Převod do HLS modelu
            Cv2.CvtColor(roi, hls, ColorConversionCodes.BGR2HLS);

            // Hledáme bílou barvu (vysoká světlost L) - pruhy jsou bílé a na obrazu jsou výrazné
            Cv2.InRange(hls, new Scalar(0, 170, 0), new Scalar(180, 255, 60), binary);

            Cv2.Canny(binary, edges, 50, 150);

            var lines = Cv2.HoughLinesP(edges, 1, Math.PI / 180, 40, 50, 80);
            if (lines == null) return;
            foreach (var l in lines)
            {
                
                double dx = l.P2.X - l.P1.X;
                double dy = l.P2.Y - l.P1.Y;
                var angle = Math.Abs(Math.Atan2(dy, dx) * 180 / Math.PI);

                // Filtr na úhel (pruhy na silnici nejsou vodorovné)
                if (angle is > 15 and < 80)
                {
                    //Nakresli linku na tom pruhu co vidíš
                    Cv2.Line(frame, new OpenCvSharp.Point(l.P1.X, l.P1.Y + startY),
                        new OpenCvSharp.Point(l.P2.X, l.P2.Y + startY), Scalar.LimeGreen, 4);
                }
            }
        }

        private Mat GetRobustRedMask(Mat hsv)
        {
            var m1 = new Mat();
            var m2 = new Mat();
            var res = new Mat();
            Cv2.InRange(hsv, new Scalar(0, 80, 60), new Scalar(12, 255, 255), m1);
            Cv2.InRange(hsv, new Scalar(160, 80, 60), new Scalar(180, 255, 255), m2);
            Cv2.BitwiseOr(m1, m2, res);
            return res;
        }

        private Mat GetRobustBlueMask(Mat hsv)
        {
            var res = new Mat();
            Cv2.InRange(hsv, new Scalar(95, 80, 50), new Scalar(135, 255, 255), res);
            return res;
        }
    }
}