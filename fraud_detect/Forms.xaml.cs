namespace fraud_detect;

public partial class Forms : ContentPage
{
	public Forms()
	{
		InitializeComponent();

        var errors = new List<ErrorModel>
            {
                new ErrorModel { ErrorName = "Bad Card Number" },
                new ErrorModel { ErrorName = "Bad CVV" },
                new ErrorModel { ErrorName = "Bad Expiration" },
                new ErrorModel { ErrorName = "Bad PIN" },
                new ErrorModel { ErrorName = "Bad Zipcode" },
                new ErrorModel { ErrorName = "Insufficient Balance" },
                new ErrorModel { ErrorName = "Technical Glitch" },
            };

        errorsListView.ItemsSource = errors;
        BindingContext = this;
    }

    private void OnSubmitButtonClicked(object sender, EventArgs e)
    {
        string user = userEntry.Text;
        string card = cardEntry.Text;
        string day = dayPicker.Date.ToString("yyyy-MM-dd");
        string time = timePicker.Time.ToString(@"hh\:mm");
        string amount = amountEntry.Text;
        string useChip = usechipPicker.SelectedItem as string;
        string name = nameEntry.Text;
        /*string city = cityEntry.Text;
        string state = stateEntry.Text;*/
        string mcc = mccEntry.Text;
        

        // Do something with the form values, e.g., send them to a server or process locally
        // You can replace this with your logic
        /*Console.WriteLine($"User: {user}, Card: {card}, Day: {day}, Time: {time}, Amount: {amount}," +
            $"Use Chip: {useChip}, Name: {name},  City: {city}, State: {state}, MCC: {mcc}");*/
        Console.WriteLine($"User: {user}, Card: {card}, Day: {day}, Time: {time}, Amount: {amount}," +
            $"Use Chip: {useChip}, Name: {name}, MCC: {mcc}");
    }

    public class ErrorModel
    {
        public string ErrorName { get; set; }
        public bool IsSelected { get; set; }
    }
}