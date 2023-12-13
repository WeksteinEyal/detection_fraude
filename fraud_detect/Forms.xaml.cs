namespace fraud_detect;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Globalization;

//using Microsoft.ML.Transforms.Onnx;

public partial class Forms : ContentPage
{
    public Forms()
	{
        InitializeComponent();    
    }


    private void OnSubmitButtonClicked(object sender, EventArgs e)
    {
        string user = userEntry.Text;
        string card = cardEntry.Text;
        string year = yearPicker.Date.ToString("yyyy");
        string time = timePicker.Time.ToString(@"hh\:mm");
        string amount = amountEntry.Text;
        string name = nameEntry.Text;
        string mcc = mccEntry.Text;
        
        System.Console.WriteLine($"User: {user}, Card: {card}, Year: {year}, Time: {time}, Amount: {amount}," +
            $"Name: {name}, MCC: {mcc}");

        //--PARTIE ONNX

        //------PREPROCESSING
        /* Ici pour que le label encoder soit pertinent il faut appliquer à des listes identiques aux colonnes qu'on a label encodé
        sur Python. Une solution plus optimale serait d'avoir des dictionnaires déjà prêt plutôt que le reconstruire à chaque fois.
        Pour l'instant je laisse comme cela avec A, B, C car les données du dataset d'entraînement sont très particulières et il est
        quasiment impossible que la donnée entrée par l'utilisateur ait été déjà encodé. Par défaut une donnée non encodé devient -1. 
        Laisser comme cela n'a donc pas vraiment d'impact dans notre contexte. */
        List<string> merchantNames = new List<string> { "A", "B", "C", "A", "B", "C" };
        Dictionary<string, int> labelMapping = new Dictionary<string, int>();
        int labelCounter = 0;
        foreach (string nn in merchantNames.Distinct())
        { labelMapping[nn] = labelCounter; labelCounter++; }
        List<int> encodedLabels = merchantNames.Select(name => labelMapping[name]).ToList();

        List<string> userNames = new List<string> { "A", "B", "C", "A", "B", "C" };
        Dictionary<string, int> labelMappingUser = new Dictionary<string, int>();
        int labelCounterUser = 0;
        foreach (string nn in userNames.Distinct())
        { labelMappingUser[nn] = labelCounterUser; labelCounterUser++; }
        List<int> encodedLabelsUser = userNames.Select(name => labelMappingUser[name]).ToList();

        List<string> cardNames = new List<string> { "A", "B", "C", "A", "B", "C" };
        Dictionary<string, int> labelMappingCard = new Dictionary<string, int>();
        int labelCounterCard = 0;
        foreach (string nn in cardNames.Distinct())
        { labelMappingCard[nn] = labelCounterCard; labelCounterCard++; }
        List<int> encodedLabelsCard = cardNames.Select(name => labelMappingCard[name]).ToList();


        //----------TIME
        string[] parts = time.Split(':');
        int hours = int.Parse(parts[0]);
        int minutes = int.Parse(parts[1]);
        int f_time = hours * 60 + minutes;

        //-----------AMOUNT
        float f_amount = float.Parse(amount, CultureInfo.InvariantCulture);

        //-----------YEAR & MCC
        int f_year = int.Parse(year);
        int f_mcc = int.Parse(mcc);

        //-----------USER & CARD & NAME
        int f_user = labelMappingUser.ContainsKey(user) ? labelMappingUser[user] : -1;
        int f_card = labelMappingCard.ContainsKey(card) ? labelMappingCard[card] : -1;
        int f_name = labelMapping.ContainsKey(name) ? labelMapping[name] : -1;

        //------Modele
        var ONNXModelPath_ = @"C:\Users\eyalw\Desktop\Cours\Projet d'étude\Scripts\Try_model_for_c#\rnn_test_3.onnx";
        var ONNXModelPath = "rnn_test_3.onnx";

        int[] dimensions = new int[] { 1, 1, 7 };
        var inputTensor = new DenseTensor<float>(new float[]
        {
                f_user, f_card, f_year, f_time, f_amount, f_name, f_mcc
        }, dimensions);
        var features_input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("input_features", inputTensor) };

        var inferenceSession = new InferenceSession(ONNXModelPath);
        var sessionOutput = inferenceSession.Run(features_input);

        var res = ((DenseTensor<float>)sessionOutput.Single().Value).ToArray();


        float threshold = 0.5f;
        string prediction = (res[0] > threshold) ? "Normale" : "Fraude";

        DisplayAlert("Result :", $"Type de transaction: {prediction}\nRisque de fraude: {res[0]}\nInfos développer (à supprimer pour le final) user : {f_user}, amount :{f_amount}, card :{f_card}, year :{f_year}, mcc :{f_mcc}, name :{f_name}, time :{f_time}", "Ok");

    }

    private void MCCOnInfoButtonClicked(object sender, EventArgs e)
    {
        DisplayAlert("MCC Information", "Merchant Category Code is a four-digit numbers used by credit cards to classify a business" +
            " or any kind of goods and services offered by a business.", "OK");
    }
    private void CardOnInfoButtonClicked(object sender, EventArgs e)
    {
        DisplayAlert("Card Information", "This is information about Card.", "OK");
    }
    private void YearOnInfoButtonClicked(object sender, EventArgs e)
    {
        DisplayAlert("Year Information", "Provide the year of the transaction.", "OK");
    }
    private void TimeOnInfoButtonClicked(object sender, EventArgs e)
    {
        DisplayAlert("Time Information", "Provide the hour of the transaction.", "OK");
    }
    private void UserOnInfoButtonClicked(object sender, EventArgs e)
    {
        DisplayAlert("User Information", "This is information about User.", "OK");
    }
    private void AmountOnInfoButtonClicked(object sender, EventArgs e)
    {
        DisplayAlert("Amount Information", "Provide the amount of the transaction.", "OK");
    }
    private void MerchantNameOnInfoButtonClicked(object sender, EventArgs e)
    {
        DisplayAlert("Merchant Name Information", "This is information about Merchant Name.", "OK");
    }
}