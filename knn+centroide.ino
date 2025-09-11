#include <Wire.h>
#include <Adafruit_TCS34725.h>
#include <EEPROM.h>

#define NUM_COLORS 3
#define SAMPLES_PER_COLOR 20
#define TOTAL_SAMPLES (NUM_COLORS * SAMPLES_PER_COLOR)
#define K_NEIGHBORS 3
#define READS_PER_SAMPLE 4
#define WAIT_BETWEEN_READS 60
#define SIGNATURE_ADDR 0
#define SIGNATURE_VALUE 0x42
#define DATA_START_ADDR 1

Adafruit_TCS34725 tcs(TCS34725_INTEGRATIONTIME_600MS, TCS34725_GAIN_1X);

struct ColorData {
  uint8_t r, g, b;
  uint8_t label; // 0=Rojo,1=Verde,2=Naranja
};
const char* names[NUM_COLORS] = {"ROJO","VERDE","NARANJA"};

// ---------------- EEPROM ----------------
bool isTrained() {
  uint8_t sig; EEPROM.get(SIGNATURE_ADDR, sig);
  return sig == SIGNATURE_VALUE;
}
void setTrainedFlag() { uint8_t sig=SIGNATURE_VALUE; EEPROM.put(SIGNATURE_ADDR, sig); }
void clearTrainedFlag(){ uint8_t sig=0xFF; EEPROM.put(SIGNATURE_ADDR, sig); }
void writeSampleToEEPROM(int idx, const ColorData &d){ EEPROM.put(DATA_START_ADDR+idx*sizeof(ColorData), d); }
ColorData readSampleFromEEPROM(int idx){ ColorData d; EEPROM.get(DATA_START_ADDR+idx*sizeof(ColorData), d); return d; }
void printEEPROMUsage() {
  int datasetBytes = TOTAL_SAMPLES * sizeof(ColorData);
  int totalBytes = datasetBytes + 1; // +1 por SIGNATURE
  Serial.print("EEPROM usada: ");
  Serial.print(totalBytes);
  Serial.print(" bytes (Dataset: ");
  Serial.print(datasetBytes);
  Serial.println(" + Firma: 1)");
}

// ---------------- NORMALIZACIÓN ----------------
bool readNormalizedRGB(int &R, int &G, int &B) {
  unsigned long sr=0, sg=0, sb=0, sc=0;
  for (int i=0;i<READS_PER_SAMPLE;i++) {
    uint16_t rr,gg,bb,cc; tcs.getRawData(&rr,&gg,&bb,&cc);
    sr+=rr; sg+=gg; sb+=bb; sc+=cc;
    delay(WAIT_BETWEEN_READS);
  }
  float r=sr/(float)READS_PER_SAMPLE, g=sg/(float)READS_PER_SAMPLE;
  float b=sb/(float)READS_PER_SAMPLE, c=sc/(float)READS_PER_SAMPLE;
  if (c<10) return false; 
  R=constrain((int)round((r/c)*255),0,255);
  G=constrain((int)round((g/c)*255),0,255);
  B=constrain((int)round((b/c)*255),0,255);
  return true;
}

// ---------------- IQR ----------------
void sortArray(int a[], int n){ for(int i=0;i<n-1;i++) for(int j=i+1;j<n;j++) if(a[i]>a[j]){int t=a[i];a[i]=a[j];a[j]=t;} }
float percentileSorted(const int arr[], int n, float p){
  float idx=p*(n-1); int i=(int)idx; float frac=idx-i;
  return (i+1<n)? arr[i]+frac*(arr[i+1]-arr[i]) : arr[i];
}
void iqrMask(const int vals[], int n, bool keep[], float &med) {
  int tmp[SAMPLES_PER_COLOR]; for(int i=0;i<n;i++) tmp[i]=vals[i];
  sortArray(tmp,n); float q1=percentileSorted(tmp,n,0.25f), q3=percentileSorted(tmp,n,0.75f);
  med=percentileSorted(tmp,n,0.5f); float iqr=q3-q1, low=q1-1.5f*iqr, high=q3+1.5f*iqr;
  for(int i=0;i<n;i++) keep[i]=(vals[i]>=low && vals[i]<=high);
}

// ---------------- ENTRENAMIENTO ----------------
void autoTrain() {
  Serial.println("ENTRENAMIENTO INICIADO");
  int base=0;
  for(uint8_t label=0; label<NUM_COLORS; label++){
    Serial.print("Coloca: "); Serial.println(names[label]); delay(4000);
    int Rs[SAMPLES_PER_COLOR], Gs[SAMPLES_PER_COLOR], Bs[SAMPLES_PER_COLOR];
    for(int i=0;i<SAMPLES_PER_COLOR;){ int R,G,B;
      if(!readNormalizedRGB(R,G,B)){ Serial.println("Lectura invalida..."); continue;}
      Rs[i]=R; Gs[i]=G; Bs[i]=B; Serial.print("Muestra "); Serial.println(i+1); i++; }
    bool kR[SAMPLES_PER_COLOR], kG[SAMPLES_PER_COLOR], kB[SAMPLES_PER_COLOR];
    float mR,mG,mB; iqrMask(Rs,SAMPLES_PER_COLOR,kR,mR); iqrMask(Gs,SAMPLES_PER_COLOR,kG,mG); iqrMask(Bs,SAMPLES_PER_COLOR,kB,mB);
    int saved=0; 
    for(int j=0;j<SAMPLES_PER_COLOR;j++) if(kR[j]&&kG[j]&&kB[j]){ ColorData d={Rs[j],Gs[j],Bs[j],label}; writeSampleToEEPROM(base+saved,d); saved++; }
    uint8_t rMed=constrain((int)round(mR),0,255), gMed=constrain((int)round(mG),0,255), bMed=constrain((int)round(mB),0,255);
    while(saved<SAMPLES_PER_COLOR){ ColorData d={rMed,gMed,bMed,label}; writeSampleToEEPROM(base+saved,d); saved++; }
    Serial.print("Guardadas "); Serial.print(saved); Serial.print(" muestras para "); Serial.println(names[label]);

    // --- cálculo e impresión del centroide de esta clase ---
    float avgR=0,avgG=0,avgB=0;
    for (int j=0;j<SAMPLES_PER_COLOR;j++) {
      ColorData d=readSampleFromEEPROM(base-SAMPLES_PER_COLOR+j);
      avgR+=d.r; avgG+=d.g; avgB+=d.b;
    }
    avgR/=SAMPLES_PER_COLOR; avgG/=SAMPLES_PER_COLOR; avgB/=SAMPLES_PER_COLOR;
    Serial.print("Centroide "); Serial.print(names[label]);
    Serial.print(": "); Serial.print(avgR,2); Serial.print(",");
    Serial.print(avgG,2); Serial.print(",");
    Serial.println(avgB,2);
    base+=SAMPLES_PER_COLOR;
  }
  setTrainedFlag(); Serial.println("ENTRENAMIENTO COMPLETADO.");
  printEEPROMUsage();
}

// ---------------- KNN ----------------
uint8_t knnClassify(const ColorData &smp, float probs[NUM_COLORS]){
  unsigned long dist[TOTAL_SAMPLES]; uint8_t labels[TOTAL_SAMPLES];
  for(int i=0;i<TOTAL_SAMPLES;i++){ ColorData d=readSampleFromEEPROM(i);
    long dr=smp.r-d.r, dg=smp.g-d.g, db=smp.b-d.b; dist[i]=dr*dr+dg*dg+db*db; labels[i]=d.label; }
  bool used[TOTAL_SAMPLES]={0}; int votes[NUM_COLORS]={0};
  for(int k=0;k<K_NEIGHBORS;k++){ unsigned long best=0xFFFFFFFF; int idx=-1;
    for(int i=0;i<TOTAL_SAMPLES;i++) if(!used[i]&&dist[i]<best){best=dist[i]; idx=i;}
    if(idx>=0){ used[idx]=true; votes[labels[idx]]++; } }
  int win=0; for(int i=1;i<NUM_COLORS;i++) if(votes[i]>votes[win]) win=i;
  for(int i=0;i<NUM_COLORS;i++) probs[i]=(votes[i]*100.0)/K_NEIGHBORS;
  return win;
}

// ---------------- SETUP / LOOP ----------------
void setup() {
  Serial.begin(9600); while(!Serial); Serial.println("Iniciando...");
  if(!tcs.begin()){ Serial.println("No se detectó TCS34725."); while(1); }
  tcs.setInterrupt(false);
  if(!isTrained()){ Serial.println("No hay dataset -> entrenar."); autoTrain(); }
  else Serial.println("Dataset listo (r=retrain, d=dump, e=EEPROM usage).");
  printEEPROMUsage();
}

void loop() {
  if(Serial.available()){ 
    char c=Serial.read();
    if(c=='r'){ clearTrainedFlag(); autoTrain(); return; }
    if(c=='d'){ 
      for(int i=0;i<TOTAL_SAMPLES;i++){ ColorData d=readSampleFromEEPROM(i);
        Serial.print(i+1); Serial.print(": "); Serial.print(d.r); Serial.print(","); Serial.print(d.g); Serial.print(","); Serial.print(d.b); Serial.print(" | "); Serial.println(names[d.label]); } 
      return; 
    }
    if(c=='e'){ printEEPROMUsage(); return; }
  }

  int R,G,B; 
  if(!readNormalizedRGB(R,G,B)){ delay(200); return; }
  ColorData smp={(uint8_t)R,(uint8_t)G,(uint8_t)B,255};
  
  float probs[NUM_COLORS];
  unsigned long t0=millis();
  uint8_t cls=knnClassify(smp,probs);
  unsigned long t1=millis()-t0;
  
  Serial.print("R:"); Serial.print(R); Serial.print(" G:"); Serial.print(G); Serial.print(" B:"); Serial.print(B);
  Serial.print(" -> Predicción: "); Serial.print(names[cls]);
  Serial.print(" | P(Rojo): "); Serial.print(probs[0]); Serial.print("%");
  Serial.print(" P(Naranja): "); Serial.print(probs[2]); Serial.print("%");
  Serial.print(" P(Verde): "); Serial.print(probs[1]); Serial.print("%");
  Serial.print(" | Tiempo: "); Serial.print(t1); Serial.println(" ms");
  Serial.println("-----------------------------");
  delay(800);
}